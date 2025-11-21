import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.projects import densepose
from detectron2.utils.visualizer import DensePoseResultsVisualizer

def get_densepose(image_path):
    cfg = get_cfg()
    densepose.add_densepose_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("densepose_rcnn_R_50_FPN_s1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("densepose_rcnn_R_50_FPN_s1x.yaml")
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(image_path)
    outputs = predictor(image)
    dp_out = outputs["instances"].get("pred_densepose", None)

    if dp_out is not None:
        vis = DensePoseResultsVisualizer(cfg, scale=1.0)
        vis_output = vis.visualize(image, outputs["instances"])
        cv2.imwrite("densepose_output.jpg", vis_output.get_image())
        print("Saved DensePose visualization.")
    else:
        print("DensePose failed to detect any person.")

if __name__ == "__main__":
    get_densepose("input.jpg")
yield