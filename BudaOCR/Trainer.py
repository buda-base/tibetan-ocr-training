import os
import json
import random
import numpy as np

from evaluate import load
from datetime import datetime
from tqdm import tqdm


# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader

from BudaOCR.Datasets import CTCDataset, ctc_collate_fn
from BudaOCR.Encoder import LabelEncoder
from BudaOCR.Networks import CTCNetwork
from BudaOCR.Models import Easter2, VanillaCRNN
from BudaOCR.Augmentations import train_transform

from BudaOCR.Utils import (
    create_dir,
    split_dataset,
    get_filename,
    shuffle_data,
)


class OCRTrainer:
    def __init__(
        self,
        network: CTCNetwork,
        label_encoder: LabelEncoder,
        train_split: float = 0.8,
        val_test_split: float = 0.5,
        image_width: int = 2000,
        image_height: int = 80,
        batch_size: int = 32,
        workers: int = 4,
        output_dir: str = "Output",
        model_name: str = "OCRModel",
        do_test_pass: bool = True,
        preload_labels: bool = False,
        is_silent: bool = False,
    ):
        self.network = network
        self.model_name = model_name
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.preload_labels = preload_labels
        self.image_width = image_width
        self.image_height = image_height
        self.label_encoder = label_encoder
        self.workers = workers

        self.cer_scorer = load("cer")
        self.do_test_pass = do_test_pass
        self.training_time = datetime.now()
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.network.optimizer, gamma=0.99)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.network.optimizer, step_size=10, gamma=0.5
        )

        self.output_dir = self._create_output_dir(output_dir)

        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size

        self.train_images = []
        self.train_labels = []

        self.valid_images = []
        self.valid_labels = []

        self.test_images = []
        self.test_labels = []

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.is_initialized = False
        self.is_silent = is_silent

        print(f"OCR-Trainer -> Architecture: {self.network.architecture}")

    def _create_output_dir(self, output_dir) -> str:
        output_dir = os.path.join(
            output_dir,
            f"{self.training_time.year}_{self.training_time.month}_{self.training_time.day}_{self.training_time.hour}_{self.training_time.minute}",
        )
        create_dir(output_dir)
        return output_dir

    def _save_dataset(self):
        out_file = os.path.join(self.output_dir, "data.distribution")

        distribution = {}
        train_data = []
        valid_data = []
        test_data = []

        for sample in self.train_images:
            sample_name = get_filename(sample)
            train_data.append(sample_name)

        for sample in self.valid_images:
            sample_name = get_filename(sample)
            valid_data.append(sample_name)

        for sample in self.test_images:
            sample_name = get_filename(sample)
            test_data.append(sample_name)

        distribution["train"] = train_data
        distribution["validation"] = valid_data
        distribution["test"] = test_data

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(distribution, f, ensure_ascii=False, indent=1)

        print(f"Saved data distribution to: {out_file}")

    def init_from_distribution(self, distribution: dict):

        self.train_images = distribution["train_images"]
        self.train_labels = distribution["train_labels"]
        self.valid_images = distribution["valid_images"]
        self.valid_labels = distribution["valid_labels"]
        self.test_images = distribution["test_images"]
        self.test_labels = distribution["test_labels"]

        print(
            f"Train Images: {len(self.train_images)}, Train Labels: {len(self.train_labels)}"
        )
        print(
            f"Validation Images: {len(self.valid_images)}, Validation Images: {len(self.valid_labels)}"
        )

        print(
            f"Test Images: {len(self.test_images)}, Test Labels: {len(self.test_labels)}"
        )
        self._save_dataset()

        self.build_datasets()
        self.get_dataloaders()

        self.is_initialized = True

    def init(
        self,
        image_paths: list[str],
        label_paths: list[str],
        train_split: float = 0.2,
        val_test_split: float = 0.5,
    ):
        images, labels = shuffle_data(image_paths, label_paths)

        (
            self.train_images,
            self.train_labels,
            self.valid_images,
            self.valid_labels,
            self.test_images,
            self.test_labels,
        ) = split_dataset(images, labels, train_split, val_test_split)

        print(
            f"Train Images: {len(self.train_images)}, Train Labels: {len(self.train_labels)}"
        )
        print(
            f"Validation Images: {len(self.valid_images)}, Validation Images: {len(self.valid_labels)}"
        )

        print(
            f"Test Images: {len(self.test_images)}, Test Labels: {len(self.test_labels)}"
        )
        self._save_dataset()

        self.build_datasets()

        min_samples = min(
            [len(self.train_images), len(self.train_images), len(self.test_images)]
        )

        if min_samples < self.batch_size:
            self.batch_size = 8
            print(
                f"Warning: Your data distribution contains samples < batch size, adjusting batch size to: {self.batch_size}"
            )

        self.get_dataloaders()

        self.is_initialized = True

    def build_datasets(self):
        if self.preload_labels:

            train_it = [k for k in self.train_labels]
            self.train_labels = [
                self.label_encoder.read_label(token) for token in tqdm(train_it)
            ]

            val_it = [k for k in self.valid_labels]
            self.valid_labels = [
                self.label_encoder.read_label(token) for token in tqdm(val_it)
            ]

            test_it = [k for k in self.test_labels]
            self.test_labels = [
                self.label_encoder.read_label(token) for token in tqdm(test_it)
            ]

        self.train_dataset = CTCDataset(
            images=self.train_images,
            labels=self.train_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width,
            augmentations=train_transform,
        )

        self.valid_dataset = CTCDataset(
            images=self.valid_images,
            labels=self.valid_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width,
        )

        self.test_dataset = CTCDataset(
            images=self.test_images,
            labels=self.test_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width,
        )

    def get_dataloaders(self):
        self.train_loader = DataLoader(
            dataset=self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        try:
            print("Checking DataLoaders..............")
            next(iter(self.train_loader))
            next(iter(self.valid_loader))
            next(iter(self.test_loader))
            print("Done!")

        except BaseException as e:
            print(f"Failed to iterate over dataset: {e}")

    def _save_checkpoint(self):
        chpt_file = os.path.join(self.output_dir, f"{self.model_name}.pth")
        checkpoint = self.network.get_checkpoint()
        torch.save(checkpoint, chpt_file)

        if not self.is_silent:
            print(f"Saved checkpoint to: {chpt_file}")

    def _save_history(self, history: dict):
        out_file = os.path.join(self.output_dir, "history.txt")

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1)

        print(f"Training history saved to: {out_file}.")

    def _save_model_config(self):
        out_file = os.path.join(self.output_dir, "model_config.json")

        print(f"Saving model config for  architecture: {self.network.architecture}")

        network_config = {
            "checkpoint": f"{self.model_name}.pth",
            "onnx-model": f"{self.model_name}.onnx",
            "architecture": self.network.architecture,
            "input_width": self.image_width,
            "input_height": self.image_height,
            "input_layer": "input",
            "output_layer": "output",
            "squeeze_channel_dim": (
                "yes" if self.network.architecture == "Easter2" else "no"
            ),
            "swap_hw": "no" if self.network.architecture == "Easter2" else "yes",
            "encoder": self.label_encoder.name,
            "charset": self.label_encoder.charset,
        }

        json_out = json.dumps(network_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        print(f"Saved model config to: {out_file}")

    def load_checkpoint(self, checkpoint_path: str):
        self.network.load_checkpoint(checkpoint_path)

    def train(
        self,
        epochs: int = 10,
        patience: int = 8,
        check_cer: bool = False,
        export_onnx: bool = True,
        silent: bool = False,
    ):
        print("Training network....")
        self.is_silent = silent

        if self.is_initialized:
            train_history = {}
            train_loss_history = []
            val_loss_history = []
            cer_score_history = []
            best_loss = None

            max_patience = patience
            current_patience = patience

            for epoch in range(epochs):
                epoch_train_loss = 0
                tot_train_count = 0

                for b_idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):  # type: ignore
                    train_loss = self.network.train_step(data)
                    epoch_train_loss += train_loss
                    tot_train_count += self.batch_size

                self.scheduler.step()
                train_loss = epoch_train_loss / tot_train_count

                if not self.is_silent:
                    avg_loss = epoch_train_loss / (b_idx + 1)
                    print(
                        f"Epoch {epoch} done, avg_loss={avg_loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.6f}"
                    )

                train_loss_history.append(train_loss)

                val_loss = self.network.evaluate(self.valid_loader, self.is_silent)

                if not self.is_silent:
                    print(
                        f"Epoch {epoch} => Val-Loss: {val_loss}, Best-loss: {best_loss}"
                    )
                val_loss_history.append(val_loss)

                if best_loss is None:
                    best_loss = val_loss
                    self._save_checkpoint()

                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_checkpoint()
                    current_patience = max_patience
                else:
                    current_patience -= 1

                    if current_patience == 0:
                        print("Early stopping training...")
                        train_history["train_losses"] = train_loss_history
                        train_history["val_losses"] = val_loss_history
                        train_history["cer_scores"] = cer_score_history

                        self._save_history(train_history)
                        self._save_model_config()

                        if export_onnx:
                            try:
                                self.network.export_onnx(
                                    self.output_dir, model_name=self.model_name
                                )
                            except BaseException as e:
                                print(f"Failed to export onnx file: {e}")

                        print("Training complete.")
                        return

                if check_cer:
                    test_data = next(iter(self.test_loader))  # type: ignore
                    test_logits, gt_labels = self.network.test(test_data)

                    idx = random.randint(0, len(test_logits) - 1)
                    pred = self.label_encoder.ctc_decode(test_logits[idx, :, :])
                    cer_score = self.cer_scorer.compute(
                        predictions=[pred], references=[gt_labels[idx]]
                    )

                    if not self.is_silent:
                        print(f"Label: {gt_labels[idx]}")
                        print(f"Prediction: {pred}")
                        print(f"CER: {cer_score}")
                    cer_score_history.append(cer_score)

            train_history["train_losses"] = train_loss_history
            train_history["val_losses"] = val_loss_history

            self._save_history(train_history)
            self._save_model_config()

            if export_onnx:
                try:
                    self.network.export_onnx(
                        self.output_dir, model_name=self.model_name
                    )
                except BaseException as e:
                    print(f"Failed to export onnx file: {e}")

            print("Training complete.")

        else:
            print(
                "Trainer was not initialized, you may want to call init() first on the trainer instance."
            )

    def evaluate(self):
        cer_scores = {}
        test_sample_idx = 0  # keeps track of the global test data index

        for _, data in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):  # type: ignore
            test_logits, gt_labels = self.network.test(data)

            for logits, label in zip(test_logits, gt_labels):
                prediction = self.label_encoder.ctc_decode(logits)

                cer_score = self.cer_scorer.compute(
                    predictions=[prediction], references=[label]
                )

                test_sample = self.test_images[test_sample_idx]
                test_sample_n = get_filename(test_sample)
                cer_scores[test_sample_n] = cer_score

                test_sample_idx += 1

        return cer_scores
