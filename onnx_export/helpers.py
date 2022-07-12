from detectron2.config import CfgNode as CN
import cv2
from pyparsing import str_type
import torch
import numpy as np
from alfred.dl.torch.common import device

from detectron2.modeling import build_model
from detectron2.data.catalog import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg

class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        print('Loaded weights from {}'.format(cfg.MODEL.WEIGHTS))

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            print("image after transform: ", image.shape)
            image = torch.as_tensor(image.astype("float32"))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

def load_test_image(f, h, w, bs=1):
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = torch.tensor(a.astype(np.float32)).to(device).unsqueeze(0).repeat(bs, 1, 1, 1)
    return a_t, a

def setup(args, config_file=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    if not config_file:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.merge_from_file(config_file)
    return cfg

def add_sparse_inst_config(cfg):
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_ON = True
    # [SparseInst]
    cfg.MODEL.SPARSE_INST = CN()

    # parameters for inference
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005
    cfg.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.45
    cfg.MODEL.SPARSE_INST.MAX_DETECTIONS = 100

    # [Encoder]
    cfg.MODEL.SPARSE_INST.ENCODER = CN()
    cfg.MODEL.SPARSE_INST.ENCODER.NAME = "FPNPPMEncoder"
    cfg.MODEL.SPARSE_INST.ENCODER.NORM = ""
    cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS = 256

    # [Decoder]
    cfg.MODEL.SPARSE_INST.DECODER = CN()
    cfg.MODEL.SPARSE_INST.DECODER.NAME = "BaseIAMDecoder"
    cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80
    # kernels for mask features
    cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128
    # upsample factor for output masks
    cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2.0
    cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False
    cfg.MODEL.SPARSE_INST.DECODER.GROUPS = 4    
    # decoder.inst_branch
    cfg.MODEL.SPARSE_INST.DECODER.INST = CN()
    cfg.MODEL.SPARSE_INST.DECODER.INST.DIM = 256
    cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS = 4
    # decoder.mask_branch
    cfg.MODEL.SPARSE_INST.DECODER.MASK = CN()
    cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256
    cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 4

    # [Loss]
    cfg.MODEL.SPARSE_INST.LOSS = CN()
    cfg.MODEL.SPARSE_INST.LOSS.NAME = "SparseInstCriterion"
    cfg.MODEL.SPARSE_INST.LOSS.ITEMS = ("labels", "masks")
    # loss weight
    cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT = 5.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT = 2.0
    # iou-aware objectness loss weight
    cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT = 1.0

    # [Matcher]
    cfg.MODEL.SPARSE_INST.MATCHER = CN()
    cfg.MODEL.SPARSE_INST.MATCHER.NAME = "SparseInstMatcher"
    cfg.MODEL.SPARSE_INST.MATCHER.ALPHA = 0.8
    cfg.MODEL.SPARSE_INST.MATCHER.BETA = 0.2

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = False

    # [Dataset mapper]
    cfg.MODEL.SPARSE_INST.DATASET_MAPPER = "SparseInstDatasetMapper"

    # [Pyramid Vision Transformer]
    cfg.MODEL.PVT = CN()
    cfg.MODEL.PVT.NAME = "b1"
    cfg.MODEL.PVT.OUT_FEATURES = ["p2", "p3", "p4"]
    cfg.MODEL.PVT.LINEAR = False

    cfg.MODEL.CSPNET = CN()
    cfg.MODEL.CSPNET.NAME = "darknet53"
    cfg.MODEL.CSPNET.NORM = ""
    # (csp-)darknet: csp1, csp2, csp3, csp4
    cfg.MODEL.CSPNET.OUT_FEATURES = ["csp1", "csp2", "csp3", "csp4"]