# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import os
import torch
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.config import get_cfg

from helpers import add_sparse_inst_config, load_test_image, DefaultPredictor
from sparseinstonnx.sparseinstonnx import SparseInstONNX # Import to register the metadata.


"""
Export the SparseInst model to ONNX format.
Test the converted model on a single image.

Command:

python3 export_onnx.py --config-file configs/coco/yolox_s.yaml --input ./images/COCO_val2014_000000002153.jpg --opts MODEL.WEIGHTS ./output/coco_yolox_s/model_final.pth

"""
torch.set_grad_enabled(False)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./onnx/data/onnx_config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="./onnx/data/test_image.png",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="verbose when onnx export",
    )
    return parser

if __name__ == "__main__":
    # Setup configs and MetaData
    args = get_parser().parse_args()
    assert os.path.isfile(args.input), "onnx export only supports using an image as input"
    cfg = setup(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # Load sample image
    h = 480
    w = 640
    inp, ori_img = load_test_image(args.input, h, w)

    # Setup model
    predictor = DefaultPredictor(cfg)
    model = predictor.model
    model = model.float()
    model.onnx_export = True

    # Export model to ONNX
    onnx_f = './onnx/output/model_converted.onnx'

    input_names, output_names, dynamic_axes = ["masks", "scores", "labels"], ["images"], {"images": {0: "batch"}}
    torch.onnx.export(
        model,
        inp,
        onnx_f,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        do_constant_folding=True,
        verbose=args.verbose,
        dynamic_axes=dynamic_axes,
    )

    # mask_pred, scores, labels = model(inp)
    # out = detr_postprocess(out, ori_img)
    # detr postprocess
    # vis_res_fast(out, ori_img, colors=colors)

    # Trace to torchscript as well
    ts_f = './onnx/output/model_converted.pt'
    traced = torch.jit.trace(model, inp)
    torch.jit.save(traced, ts_f)
