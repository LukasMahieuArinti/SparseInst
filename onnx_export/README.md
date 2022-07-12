# Installation

Install pytorch **version 1.11**, torchvision and cudatoolkit from [pytorch](https://pytorch.org/)

Install detectron2 from [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Install packages in *onnx_export/sparseinst_env.yaml*.

Install [OLive](https://github.com/microsoft/OLive) from wheel by running

    pip install onnx_export/onnxruntime_olive-0.4.0-py3-none-any.whl

# Pytorch model training

Train pytorch SparseInst model as usual but change param *'MODEL.META_ARCHITECTURE'* to *"SparseInstONNX"*

The trained model is a slightly simplified version of the original model since ONNX exporting doesn't support all pytorch modules.

# ONNX export data preparation

Copy config.yaml that was used for pytorch model training to *onnx_export/data* and rename to *onnx_config.yaml*.

Change param 'MODEL.WEIGHTS' in config to path of output pytorch model (e.g. 'output/model.pth')

# ONNX export

Run `python onnx_export/export.py` to convert pytorch model to onnx, simplify the onnx graph, and quantize the onnx model.

Resulting models and logs can be found under *onnx_export/output*.

Change optional input arguments as pleased.

# ONNX unit test

Run `python onnx_export/unittest_onnx.py` to test whether ONNX graph is well formed and whether predicted outputs of the ONNX model are close to the Pytorch model. By default, 'optimized_model.onnx' is used for these tests.

Change optional input arguments as pleased.
