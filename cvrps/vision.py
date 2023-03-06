from common import *
import cv2 as cv
import torch
from torchvision import transforms
import numpy as np

# Global variables
device = torch.device("cpu")
model: torch.jit.ScriptModule

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def load_model(path: str) -> None:
    """
    Load model from path
    :param path: Path to model
    :return: None
    """

    global model
    model = torch.load(path)
    model = torch.jit.script(model)
    model.to(device)
    model.eval()


def make_prediction(frame: np.ndarray) -> None:
    """
    Predict the most likely gesture out of rock, paper, scissors, or none
    :param frame: A frame from the webcam
    :return: None
    """

    # Convert to tensor
    frame_tensor = preprocess(frame[:, :, [2, 1, 0]]).to(device)
    output = model(frame_tensor.unsqueeze(0))
    output = torch.softmax(output[0], dim=0)
    gesture = torch.argmax(output)

    global detected_class, detected_confidence
    detected_class = CLASSES[gesture]
    detected_confidence = int(output[gesture])
