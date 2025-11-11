import os
import cv2
import torch
import json
import random
import pyewts
import pyctcdecode.decoder as CTCDecoder

from abc import ABC, abstractmethod
from evaluate import load
from datetime import datetime
from tqdm import tqdm
from typing import Optional

from albumentations.core.composition import Compose

# torch imports
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from botok import tokenize_in_stacks, normalize_unicode
from BudaOCR.Models import Easter2, VanillaCRNN
from BudaOCR.Augmentations import train_transform

from BudaOCR.Utils import (
    create_dir,
    split_dataset,
    binarize,
    preprocess_unicode,
    get_filename,
    shuffle_data,
    pad_ocr_line,
    postprocess_wylie_label,
    preprocess_unicode
)

class LabelEncoder(ABC):
    def __init__(self, charset: str | list[str], name: str):
        self.name = name
        
        if isinstance(charset, str):
            self._charset = [x for x in charset]

        elif isinstance(charset, list):
            self._charset = charset
            
        self.ctc_vocab = self._charset.copy()
        self.ctc_vocab.insert(0, " ")
        #self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)
        self.ctc_decoder = CTCDecoder.build_ctcdecoder(self.ctc_vocab)

    @abstractmethod
    def read_label(self, label_path: str):
        raise NotImplementedError
    
    @property
    def charset(self) -> list[str]:
        return self._charset
    
    @property
    def num_classes(self) -> int:
        return len(self._charset)

    def encode(self, label: str):
        enc_lbl = []
        for x in label:
            if x in self._charset:
                enc_lbl.append(self._charset.index(x)+1)
            else:
                enc_lbl.append(-1)
                print("WARNING: {x} not in charset")
        return enc_lbl

    def decode(self, inputs: list[int]) -> str:
        return "".join(self._charset[x-1] for x in inputs)
    
    def ctc_decode(self, logits):
        return self.ctc_decoder.decode(logits).replace(" ", "")
    

class StackEncoder(LabelEncoder):
    def __init__(self, charset: list[str]):
        super().__init__(charset, "stack")

    def read_label(self, label_path: str, normalize: bool = True):
        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()

        if normalize:
            label = normalize_unicode(label)
            
        label = label.replace(" ", "")
        label = preprocess_unicode(label)
        stacks = tokenize_in_stacks(label)

        return stacks
    
    def num_classes(self) -> int:
        return len(self._charset)+1


class WylieEncoder(LabelEncoder):
    def __init__(self, charset: str):
        super().__init__(charset, "wylie")
        self.converter = pyewts.pyewts()

    def read_label(self, label_path: str):
        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()
        label = preprocess_unicode(label)
        label = self.converter.toWylie(label)
        label = postprocess_wylie_label(label)

        return label
    
    def num_classes(self) -> int:
        return len(self._charset)+1


class CTCDataset(Dataset):
    def __init__(
        self,
        images: list,
        labels: list,
        label_encoder: LabelEncoder,
        img_height: int = 80,
        img_width: int = 2000,
        augmentations: Optional[Compose] = None,
    ):
        super(CTCDataset, self).__init__()

        self.images = images
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.label_encoder = label_encoder
        self.augmentations = augmentations
 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        
        if image is None:
            Exception(f"error reading image: {self.images[index]}")
            return None
            
        else:
            image = binarize(image)
                
        if self.augmentations is not None:
            aug = self.augmentations(image=image)

            image = aug["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = pad_ocr_line(
            image, target_width=self.img_width, target_height=self.img_height
        )
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        label = self.labels[index]
        target = self.label_encoder.encode(label)
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length


def ctc_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def ctc_collate_fn2(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets, target_lengths


class CTCNetwork(ABC):
    def __init__(self, model: nn.Module, ctc_type: str = "default", architecture: str = "ocr_architecture", input_width: int = 2000, input_height: int = 80) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.architecture = architecture
        self.image_height = input_height
        self.image_width = input_width
        self.num_classes = 80
        self.model = model
        self.ctc_type = ctc_type
        self.criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def full_train(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @abstractmethod
    def get_input_shape(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, checkpoint_path: str):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, data: tuple):
        raise NotImplementedError

    @abstractmethod
    def test(self, data: tuple, all_data: bool) -> tuple[list, list]:
        raise NotImplementedError
    
    def evaluate(self, data_loader, silent: bool):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), disable=silent):
            images, _, _ = [d.to(self.device) for d in data]
            with torch.no_grad():
                loss = self.forward(data)
                val_ctc_losses.append(loss / images.size(0))

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def train(
        self,
        data_batch,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        self.model.train()

        loss = self.forward(data_batch)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    def get_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return checkpoint
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def export_onnx(self, out_dir: str, model_name: str = "model", opset: int = 18) -> None:
        self.model.eval()

        model_input = torch.randn(self.get_input_shape(), device=self.device)
   
        """
        model_input = torch.randn(
            [1, 1, self.image_height, self.image_width], device=self.device
        )
        """
        out_file = f"{out_dir}/{model_name}.onnx"

        torch.onnx.export(
            self.model,
            model_input, # type: ignore
            out_file,
            export_params=True,
            opset_version=opset,
            verbose=False,
            dynamo=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print(f"Onnx file exported to: {out_file}")


class EasterNetwork(CTCNetwork):
    def __init__(
        self,
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 80,
        mean_pooling: bool = True,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005
    ) -> None:

        self.architecture = "Easter2"
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.mean_pooling = mean_pooling
        self.device = "cuda"
        self.ctc_type = ctc_type
        self.ctc_reduction = "mean" if ctc_reduction == "mean" else "sum"
        self.learning_rate = learning_rate

        self.model = Easter2(
            input_width=self.image_width,
            input_height=self.image_height,
            vocab_size=self.num_classes,
            mean_pooling=self.mean_pooling
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.criterion = nn.CTCLoss(
            blank=0,
            reduction=self.ctc_reduction,
            zero_infinity=False)
        
        self.fine_tuning = False

        super().__init__(self.model, self.ctc_type, self.architecture, self.image_width, self.image_height)

        print(f"Network -> Architecture: {self.architecture}, input width: {self.image_width}, input height: {self.image_height}")

    def get_input_shape(self):
        return [self.num_classes, self.image_height, self.image_width]


    def fine_tune(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path)
        
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
        self.load_checkpoint(checkpoint_path)

    def forward(self, data):
        images, targets, target_lengths = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        logits = self.model(images)
        logits = logits.permute(2, 0, 1)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor(
            [logits.size(0)] * batch_size
        )  # i.e. time steps
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss
    

    def test(self, data, all_data: bool = False):
        self.model.eval()

        images, targets, target_lengths = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits = self.model(images)
        logits = logits.permute(0, 2, 1)
        logits = logits.cpu().detach().numpy()
        
        target_index = 0
        if not all_data:
            sample_idx = random.randint(0, logits.shape[0]-1)

            for b_idx, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                if b_idx == sample_idx:
                    gt_label = targets[target_index:target_index+target_length+1]
                    gt_label = gt_label.cpu().detach().numpy().tolist()

                    return [logit], [gt_label]
                target_index += target_length

        else:
            gt_labels = []

            for _, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                gt_label = targets[target_index:target_index+target_length+1]
                gt_label = gt_label.cpu().detach().numpy().tolist()
                gt_labels.append(gt_label)

                target_index += target_length

            return logits, gt_labels


class CRNNNetwork(CTCNetwork):
    def __init__(
        self,
        name: str = "CRNN",
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 77,
        rnn_type: str = "lstm",
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005
    ) -> None:
        
        self.name = name
        self.architecture = "CRNN"
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.ctc_type = ctc_type
        self.ctc_reduction = ctc_reduction
        self.learning_rate = learning_rate
        self.device = "cuda"
        self.model = VanillaCRNN(
            img_width=self.image_width,
            img_height=self.image_height,
            charset_size=self.num_classes,
            rnn=self.rnn_type
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.criterion = nn.CTCLoss(
            blank=0,
            reduction=self.ctc_reduction, 
            zero_infinity=False)
        
        super().__init__(self.model, self.ctc_type, self.architecture, self.image_width, self.image_height)

    def get_input_shape(self) -> list[int]:
        return [1, 1, self.image_height, self.image_width]
    

    def fine_tune(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path)

        trainable_layers = ["conv_block_6"]

        for param in self.model.named_parameters():
        
            for train_layers in trainable_layers:
                if train_layers not in param[0]:
                    param[1].data.requires_grad = False
                else:
                    print(f"Unfreezing layer: {param[0]}")
                    param[1].data.requires_grad = True


    def forward(self, data):
        images, targets, target_lengths = [d.to(self.device) for d in data]

        logits = self.model(images)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss
    
    def test(self, data: tuple, all_data: bool = False):
        self.model.eval()

        images, targets, target_lengths = data

        images = images.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images)
        
        logits = logits.permute(1, 0, 2)
        logits = logits.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        target_index = 0

        if not all_data:
            sample_idx = random.randint(0, logits.shape[0]-1)

            for b_idx, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                if b_idx == sample_idx:
                    gt_label = targets[target_index:target_index+target_length+1]
                    gt_label = gt_label.tolist()

                    return [logit], [gt_label]
                target_index += target_length

        else:
            gt_labels = []

            for _, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                gt_label = targets[target_index:target_index+target_length+1]
                gt_label = gt_label.tolist()
                gt_labels.append(gt_label)

                target_index += target_length

            return logits, gt_labels


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
        is_silent: bool = False
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
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.network.optimizer, gamma=0.99
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

    def init(self, image_paths: list[str], label_paths: list[str], train_split: float = 0.8, val_test_split: float = 0.5):
        images, labels = shuffle_data(image_paths, label_paths)

        self.train_images, self.train_labels, self.valid_images, self.valid_labels, self.test_images, self.test_labels = split_dataset(images, labels, train_split, val_test_split)

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


        min_samples = min([len(self.train_images), len(self.train_images), len(self.test_images)])

        if min_samples < self.batch_size:
            self.batch_size = 8
            print(f"Warning: Your data distribution contains samples < batch size, adjusting batch size to: {self.batch_size}")

        self.get_dataloaders()

        self.is_initialized = True


    def build_datasets(self):
        if self.preload_labels:

            train_it = [k for k in self.train_labels]
            self.train_labels  = [self.label_encoder.read_label(token) for token in tqdm(train_it)]

            val_it = [k for k in self.valid_labels]
            self.valid_labels  = [self.label_encoder.read_label(token) for token in tqdm(val_it)]

            test_it = [k for k in self.test_labels]
            self.test_labels  = [self.label_encoder.read_label(token) for token in tqdm(test_it)]

        self.train_dataset = CTCDataset(
            images=self.train_images,
            labels=self.train_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width,
            augmentations=train_transform
        )

        self.valid_dataset = CTCDataset(
            images=self.valid_images,
            labels=self.valid_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width
        )

        self.test_dataset = CTCDataset(
            images=self.test_images,
            labels=self.test_labels,
            label_encoder=self.label_encoder,
            img_height=self.image_height,
            img_width=self.image_width
        )

    def get_dataloaders(self):
        self.train_loader = DataLoader(
            dataset=self.train_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset, # type: ignore
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
            "squeeze_channel_dim": "yes" if self.network.architecture == "Easter2" else "no",
            "swap_hw": "no" if self.network.architecture == "Easter2" else "yes",
            "encoder": self.label_encoder.name,
            "charset": self.label_encoder.charset
            
        }

        json_out = json.dumps(network_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        print(f"Saved model config to: {out_file}")

    def load_checkpoint(self, checkpoint_path: str):
        self.network.load_checkpoint(checkpoint_path)


    def train(self, epochs: int = 10, scheduler_start: int = 10, patience: int = 8, check_cer: bool = False, export_onnx: bool = True, silent: bool = False):
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

                for _, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)): # type: ignore
                    train_loss = self.network.train(data)
                    epoch_train_loss += train_loss
                    tot_train_count += self.batch_size

                train_loss = epoch_train_loss / tot_train_count
                
                if not self.is_silent:
                    print(f"Epoch {epoch} => Train-Loss: {train_loss}")
                train_loss_history.append(train_loss)

                val_loss = self.network.evaluate(self.valid_loader, self.is_silent)

                if not self.is_silent:
                    print(f"Epoch {epoch} => Val-Loss: {val_loss}, Best-loss: {best_loss}")
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
                                self.network.export_onnx(self.output_dir, model_name=self.model_name)
                            except BaseException as e:
                                print(f"Failed to export onnx file: {e}")

                        print("Training complete.")
                        return
                    
                if check_cer:
                    test_data = next(iter(self.test_loader)) # type: ignore
                    test_logits, gt_labels = self.network.test(test_data, all_data=False)

                    # that is a bit hacky, if more than 1 result is returned accumualte the results
                    gt_label = self.label_encoder.decode(gt_labels[0]) 
                    prediction = self.label_encoder.ctc_decode(test_logits[0])
        
                    if prediction != "":
                        cer_score = self.cer_scorer.compute(predictions=[prediction], references=[gt_label])

                        if not self.is_silent:
                            print(f"Label: {gt_label}")
                            print(f"Prediction: {prediction}")
                            print(f"CER: {cer_score}")
                        cer_score_history.append(cer_score)
                    else:
                        cer_score = "nan"
                        cer_score_history.append(cer_score)
                        if not self.is_silent:
                            print(f"CER: {cer_score}")

                if epoch > scheduler_start:
                    self.scheduler.step()

            train_history["train_losses"] = train_loss_history
            train_history["val_losses"] = val_loss_history

            self._save_history(train_history)
            self._save_model_config()

            if export_onnx:
                try:
                    self.network.export_onnx(self.output_dir, model_name=self.model_name)
                except BaseException as e:
                    print(f"Failed to export onnx file: {e}")

            print("Training complete.")

        else:
            print("Trainer was not initialized, you may want to call init() first on the trainer instance.")


    def evaluate(self):
        cer_scores = {}
        test_sample_idx = 0 # keeps track of the global test data index

        for _, data in tqdm(enumerate(self.test_loader), total=len(self.test_loader)): # type: ignore
            test_logits, gt_labels = self.network.test(data, all_data=True)
            
            for logits, label in zip(test_logits, gt_labels):
                gt_label = self.label_encoder.decode(label)
                prediction = self.label_encoder.ctc_decode(logits)

                cer_score = self.cer_scorer.compute(predictions=[prediction], references=[gt_label])
                
                test_sample = self.test_images[test_sample_idx]
                test_sample_n = get_filename(test_sample)
                cer_scores[test_sample_n] = cer_score

                test_sample_idx += 1

        return cer_scores


