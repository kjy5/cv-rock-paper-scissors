import argparse
from enum import Enum

# Constants
WINDOW_NAME = "CV Rock Paper Scissors"
CAM_SHAPE = [720, 1280, 3]
CAM_HEIGHT, CAM_WIDTH = CAM_SHAPE[:2]
CAM_X_START = (CAM_WIDTH - CAM_HEIGHT) // 2
CAM_X_END = CAM_WIDTH - CAM_X_START
CLASSES = ["clutter", "paper", "rock", "scissors"]

# Global variables
human_score = 0
computer_score = 0
played_class = "Scissors"


class GameState(Enum):
    """
    Game state
    """

    START = 0
    COUNT = 1
    EVAL = 2
    FAILED = 3
    RESULT = 4


def configure_argparse() -> argparse.ArgumentParser:
    """
    Configure argparse
    :return: None
    """
    parser = argparse.ArgumentParser(
        description="CV Rock Paper Scissors",
        prog="python -m cvrps",
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        dest="camera_idx",
        default=0,
        help="Video Capture Device Index (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_path",
        default="model.pth",
        help="Model path (default: model.pth)",
    )

    return parser
