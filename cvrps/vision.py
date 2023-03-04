from common import *
import cv2 as cv
import torch

# Global variables
device = torch.device("cpu")


# noinspection PyUnresolvedReferences
def configure_torch_gpu() -> None:
    """
    Configure torch to use GPU if available
    :return: None
    """
    # noinspection PyGlobalUndefined
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")


# noinspection PyUnresolvedReferences
def initialize_camera(cam_idx: int) -> cv.VideoCapture:
    """
    Initialize camera
    :return: VideoCapture device
    """

    # Initialize camera
    cam = cv.VideoCapture(cam_idx)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    print("Camera initialized")

    # Create display window
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    return cam
