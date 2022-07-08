"""
Test whether the onnx conversion went through correctly, by:
1. Checking whether the onnx graph formed correctly with onnx.checker.check_model
2. Comparing how close the individual output values of the onnx model are to the outputs of the original model
3. Comparing the predictions on an image of the onnx model to the image predictions of the original model
"""

import numpy as np
import torch
import cv2
import onnx
#import onnxruntime

from helpers import DefaultPredictor, load_test_image, setup
from sparseinstonnx.sparseinstonnx import SparseInstONNX # Import to register the metadata.
import argparse

############### First test ###############
def model_form_test(onnx_file: str) -> bool:
    """Check whether onnx model was defined correctly

    Args:
        onnx_file (str): location of onnx model

    Returns:
        bool: True or False
    """
    model = onnx.load(onnx_file)
    try:
        onnx.checker.check_model(model) # Raises exception if test fails
        return True
    except:
        return False

############### Second test ###############
def predictions_test(onnx_file: str, torch_file: str, onnx_cfg: str, torch_cfg: str, input_image: str) -> bool:
    """Test whether outputs of onnx model are close to outputs of torch model

    Args:
        onnx_file (str): location of onnx model
        torch_file (str): location of torch model
        input_image (str): location of input image

    Returns:
        bool: True or False
    """
    # Load test input image
    h = 480
    w = 640
    inp, ori_img = load_test_image(input_image, h, w)

    # Predict with pytorch model
    predictor_torch = DefaultPredictor(torch_cfg)
    model_torch = predictor_torch.model
    model_torch.onnx_export = True
    model_torch.float()
    out = model_torch(inp)
    return(out)



    # compare ONNX Runtime and PyTorch results
    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

def _onnxruntime_model_prediction(onnx_file: str, input_image:str) -> torch.Tensor:
    # Load test input image
    h = 480
    w = 640
    inp, ori_img = load_test_image(input_image, h, w)

    ort_session = onnxruntime.InferenceSession(onnx_file)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

def _pytorch_model_prediction(torch_file: str, config, input_image: str) -> torch.Tensor:
    # Load test input image
    h = 480
    w = 640
    inp, ori_img = load_test_image(input_image, h, w)


    predictor = DefaultPredictor(config).model
    predictor.onnx_export = True
    out = predictor(inp)
    print(out)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--onnx-config",
        default="./onnx/data/onnx_config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--torch-config",
        default="./onnx/data/torch_config.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--onnx-model",
        default="./onnx/output/model_converted.onnx",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--torch-model",
        default="./onnx/output/model_converted.pt",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="./onnx/data/test_image.png",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg_onnx = setup(args, args.onnx_config)

    assert model_form_test(args.onnx_model) == True, "Check onnx model test failed; onnx model is not well formed"
    _pytorch_model_prediction(args.torch_model, cfg_onnx, args.input)
    #assert predictions_test(args.onnx_model, args.torch_model, cfg_onnx, cfg_onnx, args.input) == True, "Prediction comparison failed; predictions from onnx are not close enough to torch predictions"