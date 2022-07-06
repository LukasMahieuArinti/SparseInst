# from sparseinst import add_sparse_inst_config
import os
import torch

# from detectron2.config import get_cfg
# from detectron2.engine import default_setup
from train_net import Trainer, setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch, default_argument_parser
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader,build_detection_test_loader, detection_utils
from detectron2.export import TracingAdapter, dump_torchscript_IR, scripting_with_instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.utils.env import TORCH_VERSION

from detectron2.modeling.postprocessing import detector_postprocess

import argparse

from test_net import process_batched_inputs

# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    # image = inputs[0]["image"]
    # height = torch.Tensor([inputs[0]["height"]])
    # width = torch.Tensor([inputs[0]["width"]])
    # inputs = [{"image": image, "height": height, "width": width}]  # remove other unused keys

    #input_names = ["image", "height", "width"] #instances, gt_classes, gt_masks
    # output_names = ["output1"]
    images = inputs["images"]
    resized_size = torch.Tensor(inputs["resized_size"])
    ori_size = torch.Tensor(inputs["ori_size"])
    inputs = {"images": images, "resized_size": resized_size, "ori_size": ori_size}

    inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (images, resized_size, ori_size))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (images, resized_size, ori_size), f, verbose=True)
    #logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    #logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper

def get_sample_inputs(args, cfg):

    if args.sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        images, resized_size, ori_size = process_batched_inputs(first_batch)

        return {"images": images, "resized_size": resized_size, "ori_size": ori_size}
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs

def main(args):
    cfg = setup(args)
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable respecialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)
    
    torch_model = Trainer.build_model(cfg)
    DetectionCheckpointer(torch_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True)

    torch_model.eval()

    # get sample data
    sample_inputs = get_sample_inputs(args, cfg)
    
    # export
    exported_model = export_tracing(torch_model, sample_inputs)
    print('EXPORTED', exported_model)

if __name__ == '__main__':
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument("--format", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)
