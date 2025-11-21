"""
A small interface to train OCR networks on the Easter2 or CRNN architecture. Note that the default settings were chosen with small
datasets (<20k samples in mind). Adjust the number of epochs and the patience parameter for early stopping when training on a large dataset (e.g. > 500k samples) 
to avoid unnecess
"""

import os
import argparse
import numpy as np
from BudaOCR.Config import CHARSET, STACK_FILE
from BudaOCR.Trainer import EasterNetwork, CRNNNetwork, OCRTrainer, WylieEncoder, StackEncoder
from BudaOCR.Utils import create_dir, read_stack_file, build_distribution_from_file

# disable albumentation update checks, there are issues with the latest version
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

WORKERS = 4
BATCH_SIZE = 32
CHECK_CER = True
PRELOAD_LABELS = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["easter2", "crnn"], required=False, default="easter2")
    parser.add_argument("--encoding", choices=["wylie", "stacks"], required=False, default="wylie")
    parser.add_argument("--width", type=int, required=False, default=3200)
    parser.add_argument("--height", type=int, required=False, default=80)

    parser.add_argument("--epochs", type=int, required=False, default=40)
    parser.add_argument("--export_onnx", choices=["yes", "no"], required=False, default="yes")

    args = parser.parse_args()
    input_dir = args.input
    chkpt_dir = args.checkpoint

    if args.encoding == "wylie":
        encoder = wylie_encoder = WylieEncoder(CHARSET)
    else:
        stacks = read_stack_file(STACK_FILE)
        encoder = StackEncoder(stacks)

    input_width = args.width
    input_height = args.height
    epochs = args.epochs
    export_onnx = args.export_onnx

    distr_file = f"{chkpt_dir}/data.distribution"

    assert os.path.isdir(input_dir)
    assert os.path.isdir(chkpt_dir)
    assert os.path.isfile(distr_file)

    distribution = build_distribution_from_file(distr_file, input_dir)

    output_dir = os.path.join(input_dir, "Output")
    create_dir(output_dir)

    num_classes = encoder.num_classes()

    if args.model == "easter2":
        network = EasterNetwork(num_classes=num_classes, image_width=input_width, image_height=input_height, mean_pooling=True)
    else:
        network = CRNNNetwork(image_width=input_width, image_height=input_height, num_classes=num_classes)

    ocr_trainer = OCRTrainer(
        network=network,
        label_encoder=encoder,
        workers=WORKERS,
        image_width=input_width,
        image_height=input_height,
        batch_size=BATCH_SIZE,
        output_dir=output_dir,
        preload_labels=PRELOAD_LABELS
        )

    ocr_trainer.init_from_distribution(distribution)

    if epochs > 4:
        scheduler_start = int(epochs * 0.75)

    elif epochs < 4 and epochs > 1:
        scheduler_start = epochs - 1
    
    else:
        scheduler_start = epochs + 1 # effectively ignoring it

    ocr_trainer.train(epochs=epochs, scheduler_start=scheduler_start, check_cer=CHECK_CER, export_onnx=True, silent=True)

    cer_scores = ocr_trainer.evaluate()

    cer_values = list(cer_scores.values())

    print(f"Mean CER: {np.mean(cer_values)}")
    print(f"Max CER: {np.max(cer_values)}")
    print(f"Min CER: {np.min(cer_values)}")

    score_file = os.path.join(ocr_trainer.output_dir, "cer_scores.txt")

    with open(score_file, "w", encoding="utf-8") as f:
        for sample, value in cer_scores.items():
            f.write(f"{sample} - {value}\n")