"""
Test whether the onnx conversion went through correctly, by:
1. Checking whether the onnx graph formed correctly with onnx.checker.check_model
2. Comparing how close the individual output values of the onnx model are to the outputs of the original model
"""

import numpy as np
import torch
import onnx
import onnxruntime

from helpers import DefaultPredictor, load_test_image, setup
from sparseinstonnx.sparseinstonnx import SparseInstONNX # Import to register the metadata.
import argparse
from detectron2.utils.logger import setup_logger


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
        logger.debug('ONNX model was defined correctly')
        return True
    except onnx.checker.ValidationError as e:
        logger.debug(e)
        return e

############### Second test ###############
def prediction_test(torch_prediction: list, onnx_prediction: list) -> bool:
    assert len(torch_prediction) == len(onnx_prediction), "Torch prediction and onnx prediction don't have same shape"
    logger.debug('Torch prediction and onnx prediction have same shape')
    # Test predicted probabilities
    assert np.allclose(torch_prediction[1].cpu().detach().numpy(), onnx_prediction[1], atol = 0.1), "Some predicted probabilities are not close enough"
    logger.debug('Torch and onnx predicted class probabilities are within 0.1 of each other')

############### Helpers ###############
def _onnxruntime_model_prediction(onnx_file: str, input_image:torch.Tensor) -> torch.Tensor:
    ort_session = onnxruntime.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])

    # compute ONNX Runtime output prediction
    input_name = ort_session.get_inputs()[0].name
    pred_onnx = ort_session.run(None, {input_name: input_image.cpu().numpy()})
    return(pred_onnx)

def _pytorch_model_prediction(config, input_image: torch.Tensor) -> torch.Tensor:
    predictor = DefaultPredictor(config).model
    predictor.onnx_export = True
    pred_torch = predictor(input_image)
    return(pred_torch)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--onnx-config",
        default="./onnx_export/data/onnx_config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--onnx-model",
        default="./onnx_export/output/optimized_model.onnx",
        metavar="FILE",
        help="path to onnx model",
    )
    parser.add_argument(
        "--torch-model",
        default="./onnx_export/output/model_converted.pt",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="./onnx_export/data/test_image.jpg",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    return parser

if __name__ == "__main__":
    # Load configs
    args = get_parser().parse_args()
    cfg = setup(args, args.onnx_config)

    logger = setup_logger(output = "onnx_export/output/log.txt",name="detectron2")
    logger.debug(f"UNIT TESTING ONNX MODEL")
    logger.debug("Arguments: " + str(args))

    # Load data
    h = 550
    w = 550
    input_image, _ = load_test_image(args.input, h, w)
    
    # First test
    assert model_form_test(args.onnx_model) == True, "Check onnx model test failed; onnx model is not well formed"

    # Second test
    torch_prediction = _pytorch_model_prediction(cfg, input_image)
    onnx_prediction = _onnxruntime_model_prediction(args.onnx_model, input_image)
    prediction_test(torch_prediction, onnx_prediction)
    
    logger.debug('All tests passed')
