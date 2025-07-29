"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import openvino as ov

def main(
    args,
):
    """main"""
    if not args.dynamic and args.batch_size <= 1:
        inp = None
    elif args.batch_size > 1 and args.dynamic:
        inp = [-1, 3, -1, -1]
    elif args.batch_size > 1:
        inp = [-1, 3, args.input_size(0), args.input_size(1)]
    elif args.dynamic:
        inp = [1, 3, -1, -1]

    model = ov.convert_model(
        input_model=str(args.model_path),
        input=inp,
    )
    ov.serialize(model, str(Path(args.model_path).with_suffix(".xml")), str(Path(args.model_path).with_suffix(".bin")))
    print("OpenVINO model exported")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dynamic",
        "-d",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--img_size",
        "-i",
        default=(640,640),
        type=tuple,
    )
    args = parser.parse_args()
    main(args)
