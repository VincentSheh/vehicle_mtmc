from pathlib import Path
import torch
import os
from ultralytics import YOLO


def load_yolo(which):
    """Load a yolo network from local repository. Download the weights there if needed."""

    cwd = Path.cwd()
    yolo_dir = str(Path(__file__).parent.joinpath("yolov5"))
    os.chdir(yolo_dir)
    if which.startswith("yolov5"): 
        model = torch.hub.load(yolo_dir, which, source="local")
    else:
        model = YOLO(which)
    os.chdir(str(cwd))
    return model
