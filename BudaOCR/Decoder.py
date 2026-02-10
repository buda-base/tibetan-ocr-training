from numpy.typing import NDArray

from pyctcdecode import build_ctcdecoder
from pyctcdecode.decoder import OutputBeam

from BudaOCR.Data import KenLMConfig


class CTCDecoder:
    def __init__(
        self,
        charset: str | list[str],
        kenlm_config: KenLMConfig | None,
    ):
        self.blank_sign = " "
        self.ctc_beam_width = 64

        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        self.ctc_vocab.insert(0, " ")

        if kenlm_config is not None:
            try:
                self.ctc_decoder = build_ctcdecoder(
                    self.ctc_vocab,
                    kenlm_model_path=str(kenlm_config.kenlm_file),
                    unigrams=kenlm_config.unigrams,
                )
            except Exception as e:
                print(f"KenLM disabled: {e}")
                self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)
        else:
            self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def encode(self, label: str):
        return [self.charset.index(x) + 1 for x in label]

    def decode(self, inputs: list[int]) -> str:
        return "".join(self.charset[x - 1] for x in inputs)

    def ctc_decode(self, logits) -> str:
        return self.ctc_decoder.decode(logits).replace(self.blank_sign, "")

    def ctc_beam_decode(self, logits: NDArray) -> list[OutputBeam]:
        return self.ctc_decoder.decode_beams(logits)

    def set_alpha_beta(self, alpha: float, beta: float):
        self.init_ctc_decoder(alpha, beta)
