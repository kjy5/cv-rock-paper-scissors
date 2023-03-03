import argparse
import cv2 as cv
import numpy as np
import torch

# Constants
WINDOW_NAME = "CV Rock Paper Scissors"

# Global variables
device = torch.device("cpu")
args: argparse.Namespace


def configure_torch_gpu() -> None:
    """
    Configure torch to use GPU if available
    :return: None
    """
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")


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
        help='Video Capture Device Index (default: 0)',
    )

    return parser


# noinspection PyUnresolvedReferences
def initialize_camera() -> (cv.VideoCapture, list):
    """
    Initialize camera
    :return: VideoCapture device
    """

    # Initialize camera
    cap = cv.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print("Camera initialized")

    # Create display window
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        exit()

    return cap, frame.shape


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
    cam, shape = initialize_camera()
    frame_height = shape[0]
    frame_width_start = (shape[1] - shape[0]) // 2
    frame_width_end = shape[1] - frame_width_start

    while cam.isOpened():
        # Capture frame-by-frame
        ret, frame = cam.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Save copy for processing
        to_process_frame = np.copy(
            frame[:frame_height, frame_width_start:frame_width_end])

        # Draw UI
        # Flip to help with visuals
        cv.flip(frame, 1, frame)

        # Draw a rectangle around the frame
        cv.rectangle(frame, (0, 0), (frame_width_start, shape[0]), (15, 15, 15), -1)
        cv.rectangle(frame, (frame_width_end, 0), (shape[1], shape[0]), (15, 15, 15),
                     -1)

        # Display the full frame
        cv.imshow(WINDOW_NAME, frame)

        # Press Q on keyboard to exit
        if cv.waitKey(1) == ord('q'):
            break

    # Cleanup
    print("Cleaning up")
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
