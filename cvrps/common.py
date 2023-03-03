import argparse
import torch

# Constants
WINDOW_NAME = "CV Rock Paper Scissors"
CAM_SHAPE = [720, 1280, 3]
CAM_HEIGHT, CAM_WIDTH = CAM_SHAPE[:2]
CAM_X_START = (CAM_WIDTH - CAM_HEIGHT) // 2
CAM_X_END = CAM_WIDTH - CAM_X_START

# Global variables
device = torch.device("cpu")
args: argparse.Namespace

human_score = 0
detected_class = "xxxxxxxx"
computer_score = 0
played_class = "xxxxxxxx"
