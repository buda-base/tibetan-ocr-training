import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod
from numpy.typing import NDArray

import torch
import torch.nn.functional as F

from torch import nn
from torch.amp.grad_scaler import GradScaler

from BudaOCR.Data import VitConfig
from BudaOCR.Models import ConvFrontEnd, Easter2, Easter2Attention, Easter2PlusViT, Easter2b, VanillaCRNN


class CTCNetwork(ABC):
    def __init__(
        self,
        model: nn.Module,
        architecture: str = "ocr_architecture",
        input_width: int = 2000,
        input_height: int = 80,
        num_classes: int = 80,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005,
    ) -> None:

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_str = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_str = "cpu"

        self.architecture = architecture
        self.model = model.to(self.device)

        self.image_height = input_height
        self.image_width = input_width
        self.num_classes = num_classes

        self.ctc_type = ctc_type
        self.criterion = nn.CTCLoss(
            blank=0, reduction=ctc_reduction, zero_infinity=True
        )
        self.amp = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler(self.device_str) if self.amp else None

        self.fine_tuning = False

    def full_train(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @abstractmethod
    def get_input_shape(self) -> list[int]:
        raise NotImplementedError
    

    def reset_optimizer(self):
        self.optimizer.state.clear()

    @abstractmethod
    def fine_tune(self, checkpoint_path: str):
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: tuple, scaler: GradScaler | None):
        raise NotImplementedError

    @abstractmethod
    def test(self, data: tuple):
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self,
        data_batch: torch.Tensor,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        raise NotImplementedError

    def get_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return checkpoint

    def evaluate(self, data_loader, silent: bool):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(
            enumerate(data_loader), total=len(data_loader), disable=silent
        ):
            images, _, _, _ = data
            images = images.to(self.device)

            with torch.no_grad():
                loss = self.forward(data, self.scaler)
                val_ctc_losses.append(loss / images.size(0))

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def load_checkpoint(self, checkpoint_path: str, device: str = "cuda"):

        if device == "cpu":
            map_location=torch.device('cpu')
            checkpoint = torch.load(checkpoint_path, map_location)

        # assuming CUDA by default
        else:
            checkpoint = torch.load(checkpoint_path)
            
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def export_onnx(
        self, out_dir: str, model_name: str = "model", opset: int = 18
    ) -> None:
        self.model.eval()

        model_input = torch.randn(self.get_input_shape(), device=self.device)
        print(f"Exporting ONNX for model input shape: {model_input.shape}")
        """
        model_input = torch.randn(
            [1, 1, self.image_height, self.image_width], device=self.device
        )
        """
        out_file = f"{out_dir}/{model_name}.onnx"

        torch.onnx.export(
            self.model,
            model_input,
            out_file,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"}
            },
            do_constant_folding=False,
        )

        self.model.to(self.device)
        print(f"Onnx file exported to: {out_file}")

"""
CRNN
"""

class CRNNNetwork(CTCNetwork):
    def __init__(
        self,
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 77,
        rnn_type: str = "lstm",
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005,
    ) -> None:

        model = VanillaCRNN(
            img_width=image_width,
            img_height=image_height,
            charset_size=num_classes,
            rnn=rnn_type,
        )

        super().__init__(
            model,
            "CRNN",
            image_width,
            image_height,
            num_classes,
            ctc_type,
            ctc_reduction,
            learning_rate,
        )

    def get_input_shape(self) -> list[int]:
        return [1, 1, self.image_height, self.image_width]

    def fine_tune(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path, self.device_str)

        trainable_layers = ["conv_block_6"]

        for param in self.model.named_parameters():
            for train_layers in trainable_layers:
                if train_layers not in param[0]:
                    param[1].data.requires_grad = False
                else:
                    print(f"Unfreezing layer: {param[0]}")
                    param[1].data.requires_grad = True

    def forward(self, data, scaler: GradScaler):
        images, targets, target_lengths, _ = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        logits = self.model(images)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss

    def train_step(
        self,
        data_batch,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        self.model.train()

        loss = self.forward(data_batch, self.scaler)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    def test(self, data: tuple, all_data: bool = False):
        self.model.eval()

        images, targets, target_lengths, gt_labels = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits = self.model(images)

        np_logits = logits.detach().cpu().numpy()
        np_logits = np.transpose(np_logits, axes=[1, 0, 2])

        print(f"CRNN Logits: {np_logits.shape}")
        # B = Batch dim, T=Time dim, V=Vocabulary=num_classes
        # np_logits = np.transpose(np_logits, axes=[0, 2, 1]) # BxTxV

        return np_logits, gt_labels


"""
Easter2 (original)
"""

class EasterNetwork(CTCNetwork):
    def __init__(
        self,
        variant: str = "Easter2",
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 80,
        mean_pooling: bool = True,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005,
    ) -> None:

        self.mean_pooling = mean_pooling
        self.learning_rate = learning_rate

        if variant == "Easter2":
            self.model = Easter2(
                input_channels=image_height,
                vocab_size=num_classes,
                mean_pooling=mean_pooling,
            )
        elif variant == "Easter2b":
            self.model = Easter2b(image_height, vocab_size=num_classes)

        else:
            raise ValueError("Undefined Easter2 variant provided")
        

        super().__init__(
            self.model,
            variant,
            image_width,
            image_height,
            num_classes,
            ctc_type,
            ctc_reduction,
            learning_rate,
        )

        print(
            f"Network -> Architecture: {self.architecture}, input width: {self.image_width}, input height: {self.image_height}"
        )

    def get_input_shape(self):
        return [self.num_classes, self.image_height, self.image_width]

    def fine_tune(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path, self.device_str)

        trainable_layers = ["conv1d_5"]

        for param in self.model.named_parameters():

            for train_layers in trainable_layers:
                if train_layers not in param[0]:
                    param[1].data.requires_grad = False
                else:
                    if "easter" not in param[0]:
                        print(f"Unfreezing layer: {param[0]}")
                        param[1].data.requires_grad = True

        self.fine_tuning = True

    def load_model(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path, self.device_str)

    def forward(self, data, scaler: GradScaler):
        images, targets, target_lengths, _ = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        logits, _ = self.model(images)
        logits = logits.permute(2, 0, 1)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor(
            [logits.size(0)] * batch_size
        )  # i.e. time steps
        target_lengths = torch.flatten(target_lengths)

        return self.criterion(log_probs, targets, input_lengths, target_lengths)

    def train_step(
        self,
        data_batch: torch.Tensor,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        self.model.train()

        loss = self.forward(data_batch, self.scaler)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    def test(self, data: tuple) -> tuple[NDArray, list[str]]:
        images, targets, target_lengths, gt_labels = data

        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(images)

        np_logits = logits.detach().cpu().numpy()
        # B = Batch dim, T=Time dim, V=Vocabulary=num_classes
        np_logits = np.transpose(np_logits, axes=[0, 2, 1])  # BxTxV

        return np_logits, gt_labels


class Easter2AttNetwork(CTCNetwork):
    """
    A modified Network architecture that uses a modified Easter2 version (Easter2b) as backbone together with a light Attention Head
    """

    def __init__(
        self,
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 80,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005,
        easter_variant: str = "default",
    ) -> None:

        model = Easter2Attention(
            vocab_size=num_classes,
            input_height=image_height,
            easter_variant=easter_variant,
        )

        super().__init__(
            model,
            "Easter2Attention",
            image_width,
            image_height,
            num_classes,
            ctc_type,
            ctc_reduction,
            learning_rate,
        )

    def get_input_shape(self):
        return [self.num_classes, self.image_height, self.image_width]

    def forward(self, data: GradScaler, scaler: bool):
        images, targets, target_lengths, _ = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        if scaler is not None:
            with torch.amp.autocast(self.device_str, dtype=torch.float16, enabled=True):
                logits = self.model(images)
            # outputs may be (main, aux) or single tensor
        else:
            logits = self.model(images)

        with torch.amp.autocast(self.device_str, enabled=False):
            if isinstance(logits, (list, tuple)):
                main_logits, _ = logits[0], logits[1]
            else:
                main_logits, _ = logits, None

            # main_logits: (B, V, T)
            # CTC expects (T, N, C) -> so permute and take log_softmax across C
            main_logits = main_logits.float()
            log_probs = main_logits.log_softmax(dim=1)  # (B, V, T)
            log_probs = log_probs.permute(2, 0, 1)  # (T, B, V)

            # compute input_lengths: model-specific. We approximate by T for each batch (no downsampling info)
            T_seq = log_probs.size(0)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=T_seq, dtype=torch.long
            ).to(self.device)
            target_lengths = torch.flatten(target_lengths)

            return self.criterion(log_probs, targets, input_lengths, target_lengths)

    def fine_tune(self, checkpoint_path: str): # TODO
        pass

    def train_step(
        self,
        data_batch: torch.Tensor,
        clip_grads: bool = True, # TODO
        grad_clip: float = 5.0,
    ):
        self.model.train()
        loss = self.forward(data_batch, self.scaler)
        grad_clip = 5.0

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self, data_loader, silent: bool):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(
            enumerate(data_loader), total=len(data_loader), disable=silent
        ):
            images, _, _, _ = data
            images = images.to(self.device)
            with torch.no_grad():
                loss = self.forward(data, self.scaler)
                val_ctc_losses.append(loss.item())

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def test(self, data: tuple) -> tuple[NDArray, list[str]]:
        images, targets, target_lengths, gt_labels = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits = self.model(images)

        np_logits = logits.detach().cpu().numpy()
        # B = Batch dim, T=Time dim, V=Vocabulary=num_classes
        np_logits = np.transpose(np_logits, axes=[0, 2, 1])  # BxTxV

        return np_logits, gt_labels

    def export_onnx(
        self, out_dir: str, model_name: str = "model", opset: int = 18
    ) -> None:

        cpu_device = torch.device("cpu")
        self.model.to(cpu_device)
        self.model.eval()

        model_input = torch.randn(
            [1, 1, self.image_height, self.image_width], device=cpu_device
        )

        out_file = f"{out_dir}/{model_name}.onnx"

        torch.onnx.export(
            self.model,
            model_input,
            out_file,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={
                "x": {0: "batch"}
            },
            do_constant_folding=False,
        )

        self.model.to(self.device)
        print(f"Exported ONNX model to {out_file}")

"""
An Easter2 variant with ViT.
"""

class Easter2ViTNetwork(CTCNetwork):
    """
    A modified Network architecture that uses Easter2 as backbone.
    """

    def __init__(
        self,
        vit_cfg: VitConfig,
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 80,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005,
        easter_variant="fixed",
    ) -> None:

        assert vit_cfg is not None

        cnn = ConvFrontEnd(out_ch=64, input_height=image_height)
        backbone = Easter2b(input_height=64*(image_height//4), vocab_size=num_classes)
        model = Easter2PlusViT(cnn, backbone, vit_cfg, vocab_size=num_classes)

        super().__init__(
            model,
            "Easter2PlusVit",
            image_width,
            image_height,
            num_classes,
            ctc_type,
            ctc_reduction,
            learning_rate,
        )

    def get_input_shape(self):
        return [self.num_classes, self.image_height, self.image_width]

    def forward(self, data: torch.Tensor, scaler: GradScaler):
        images, targets, target_lengths, _ = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        log_probs = self.model(images)

        with torch.amp.autocast(self.device_str, enabled=True):
            if isinstance(log_probs, (list, tuple)):
                out, _ = log_probs[0], log_probs[1]
            else:
                out = log_probs
            # main_logits: (B, V, T)
            # CTC expects (T, N, C) -> so permute and take log_softmax across C
            #main_logits = main_logits.float()
            #log_probs = main_logits.log_softmax(dim=1)  # (B, V, T)
            out = log_probs.permute(2, 0, 1)  # (T, B, V)

            # compute input_lengths: model-specific. We approximate by T for each batch (no downsampling info)
            T_seq = out.size(0)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=T_seq, dtype=torch.long
            ).to(self.device)
            target_lengths = torch.flatten(target_lengths)

            return self.criterion(out, targets, input_lengths, target_lengths)

    def fine_tune(self, checkpoint_path: str):
        pass

    def train_step(
        self,
        data_batch: torch.Tensor,
        clip_grads: bool = True,
        grad_clip: float = 5.0,
    ):
        self.model.train()
        loss = self.forward(data_batch, self.scaler)
        grad_clip = 5.0

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self, data_loader, silent: bool):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(
            enumerate(data_loader), total=len(data_loader), disable=silent
        ):
            images, _, _, _ = data
            images = images.to(self.device)
            with torch.no_grad():
                loss = self.forward(data, self.scaler)
                val_ctc_losses.append(loss.item())

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def test(self, data: tuple) -> tuple[NDArray, list[str]]:
        images, targets, target_lengths, gt_labels = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits = self.model(images)

        np_logits = logits.detach().cpu().numpy()
        # B = Batch dim, T=Time dim, V=Vocabulary=num_classes
        np_logits = np.transpose(np_logits, axes=[0, 2, 1])  # BxTxV

        return np_logits, gt_labels

    def export_onnx(
        self, out_dir: str, model_name: str = "model", opset: int = 18
    ) -> None:

        device = torch.device("cpu")
        self.model.to(device)
        self.model.eval()

        _, input_height, input_width = self.get_input_shape()
        model_input = torch.randn(
            1, 1, input_height, input_width,
            dtype=torch.float32,
            device="cpu"
        )

        """
        model_input = torch.randn(
            [1, 1, self.image_height, self.image_width], device=self.device
        )
        """
        out_file = f"{out_dir}/{model_name}.onnx"

        torch.onnx.export(
            self.model,
            model_input,
            out_file,
            opset_version=18,
            input_names=["input"],
            output_names=["logits"],
            dynamic_shapes={
                "x": {0: "batch"}
            },
            do_constant_folding=False,
        )

        self.model.to(self.device)
        print(f"Exported ONNX model to {out_file}")
