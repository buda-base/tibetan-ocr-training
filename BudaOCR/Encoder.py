import pyewts
import pyctcdecode.decoder as CTCDecoder

from abc import ABC, abstractmethod
from botok import normalize_unicode, tokenize_in_stacks

from BudaOCR.Utils import (
    postprocess_wylie_label,
    preprocess_unicode,
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
        self.ctc_decoder = CTCDecoder.build_ctcdecoder(self.ctc_vocab)

    @abstractmethod
    def read_label(self, label_path: str):
        raise NotImplementedError

    @property
    def charset(self) -> list[str]:
        return self._charset

    @property
    def concat_charset(self) -> str:
        return "".join(x for x in self._charset)

    @property
    def num_classes(self) -> int:
        return len(self._charset)

    def encode(self, label: str):
        enc_lbl = []
        for x in label:
            if x in self._charset:
                enc_lbl.append(self._charset.index(x) + 1)
            else:
                enc_lbl.append(-1)
                print(f"WARNING: {x} not in charset")
        return enc_lbl

    def decode(self, inputs: list[int]) -> str:
        return "".join(self._charset[x - 1] for x in inputs)

    def ctc_decode(self, logits) -> str | list[str]:
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
    
    @property
    def num_classes(self) -> int:
        return len(self._charset) + 1


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

    @property
    def num_classes(self) -> int:
        return len(self._charset) + 1
