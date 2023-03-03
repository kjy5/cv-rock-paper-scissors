import argparse
import cv2 as cv
import numpy as np
import torch
from common import *
from ui import draw_ui


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

    return parser


# noinspection PyUnresolvedReferences
def initialize_camera() -> cv.VideoCapture:
    """
    Initialize camera
    :return: VideoCapture device
    """

    # Initialize camera
    cam = cv.VideoCapture(args.camera_idx)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    print("Camera initialized")

    # Create display window
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    return cam


def main() -> None:
    """
    Run program
    :return: None
    """

    # Extract arguments from CLI
    global args
    args = configure_argparse().parse_args()

    # Setup PyTorch
    configure_torch_gpu()

    # Initialize camera and compute square crop
    cam = initialize_camera()

    while cam.isOpened():
        # Capture frame-by-frame
        ret, frame = cam.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize to 720p
        frame = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        # Save copy for processing
        to_process_frame = np.copy(frame[:CAM_HEIGHT, CAM_X_START:CAM_X_END])

        # Draw UI
        draw_ui(frame)

        # Display the full frame
        cv.imshow(WINDOW_NAME, frame)

        # Press Q on keyboard to exit
        if cv.waitKey(1) == ord("q"):
            break

    # Cleanup
    print("Cleaning up")
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
