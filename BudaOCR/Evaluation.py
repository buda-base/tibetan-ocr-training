import math
import torch
import numpy as np

from evaluate import load
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from BudaOCR.Data import Encoding, KenLMConfig, VitConfig
from BudaOCR.Datasets import CTCDataset, ctc_collate_fn
from BudaOCR.Encoder import StackEncoder, WylieEncoder
from BudaOCR.Networks import CTCNetwork, Easter2ViTNetwork, EasterNetwork
from BudaOCR.Utils import get_filename

from BudaOCR.Data import CTCModelConfig
from BudaOCR.Decoder import CTCDecoder


class Evaluator:
    """
    A simple wrapper class around some inference functions ro run ocr inference and CER calculation
    based on line-image and line-label inputs
    """

    def __init__(
        self,
        model_config: CTCModelConfig,
        device: str,
        kenlm_config: KenLMConfig | None = None,
    ):
        self.model_config = model_config
        self.device = device
        self.kenlm_config = kenlm_config
        self.cer_scorer = load("cer")

        if self.model_config.encoding == Encoding.WYLIE:
            self.label_encoder = WylieEncoder(self.model_config.charset)
        else:
            assert self.model_config.encoding == Encoding.UNICODE
            self.label_encoder = StackEncoder(self.model_config.charset)

        if kenlm_config is not None:
            self.ctc_decoder = CTCDecoder(self.model_config.charset, kenlm_config)
        else:
            self.ctc_decoder = CTCDecoder(self.model_config.charset, kenlm_config=None)

        
        if self.model_config.architecture == "Easter2PlusVit":
            print("Using Easter2-ViT")
            vit_cfg = VitConfig()

            self.network = Easter2ViTNetwork(
                vit_cfg,
                self.model_config.input_width,
                self.model_config.input_height,
                num_classes=self.label_encoder.num_classes,
            )

        elif model_config.architecture == "Easter2b":

            self.network = EasterNetwork(
                variant="Easter2b",
                image_width=self.model_config.input_width,
                image_height=self.model_config.input_height,
                num_classes=self.label_encoder.num_classes,
            )

        self.network.load_checkpoint(self.model_config.checkpoint, self.device)

    def _build_dataloader(
        self, images: list[str], labels: list[str], batch_size: int, num_workers: int
    ) -> DataLoader:
        dataset = CTCDataset(
            images,
            labels,
            self.label_encoder,
            self.model_config.input_height,
            self.model_config.input_width,
        )

        dataloader = DataLoader(
            dataset=dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

        return dataloader

    def evaluate(
        self,
        images: list[str],
        labels: list[str],
        num_workers: int = 4,
        batch_size: int = 8,
    ) -> list:
        _it = [k for k in labels]
        eval_labels = [self.label_encoder.read_label(token) for token in tqdm(_it)]

        dataloader = self._build_dataloader(
            images, eval_labels, batch_size, num_workers
        )

        results = []

        for sample_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):  # type: ignore
            logits, gt_labels = self.network.test(data)

            for b_idx in range(logits.shape[0]):
                beams = self.ctc_decoder.ctc_beam_decode(logits[b_idx])
                pred = beams[0].text.strip().replace(" ", "")

                if isinstance(gt_labels[b_idx], list):
                    gt_label = "".join(x for x in gt_labels[b_idx])
                else:
                    gt_label = gt_labels[b_idx]

                cer_score = self.cer_scorer.compute(
                    predictions=[pred], references=[gt_label]
                )

                L = max(len(beams[0].text), 1)
                norm_logp = beams[0].logit_score / L
                ctc_conf = float(math.exp(norm_logp))

                global_idx = sample_idx * batch_size + b_idx
                img_id = images[global_idx]
                img_id = Path(img_id).stem

                score_set = {
                    "img": str(img_id),
                    "pred": pred,
                    "label": gt_label,
                    "cer": cer_score,
                    "ctc_conf": ctc_conf,
                }

                results.append(score_set)

        return results

    def export_logits(
        self,
        images: list[str],
        labels: list[str],
        out_dir: str = "Output",
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        _it = [k for k in labels]
        eval_labels = [self.label_encoder.read_label(token) for token in tqdm(_it)]
        dataloader = self._build_dataloader(
            images, eval_labels, batch_size, num_workers
        )

        for sample_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            logits, gt_labels = self.network.test(data)

            for b_idx in range(logits.shape[0]):

                global_idx = sample_idx * batch_size + b_idx
                sample_id = get_filename(images[global_idx])

                np.savez_compressed(
                    out_path / f"{sample_id}.npz",
                    logits=logits[b_idx],
                    gt_text=gt_labels[b_idx],
                    sample_idx=sample_idx,
                )
