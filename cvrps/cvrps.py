import argparse
import cv2 as cv
import numpy as np
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
computer_score = 0


# noinspection PyUnresolvedReferences
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


def draw_ui(frame: np.ndarray) -> None:
    """
    Draw UI
    :param frame: Webcam frame to draw on
    :return: None
    """
    # Flip to help with visuals
    cv.flip(frame, 1, frame)

    # Variables
    margin = int(0.02 * CAM_WIDTH)

    # Human side
    cv.rectangle(frame, (0, 0), (CAM_X_START, CAM_HEIGHT), (15, 15, 15), -1)

    # Title
    cv.putText(
        frame,
        "You",
        (margin, int(.1 * CAM_HEIGHT)),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
        cv.LINE_AA,
    )
    cv.line(
        frame,
        (margin, int(0.15 * CAM_HEIGHT)),
        (CAM_X_START - margin, int(0.15 * CAM_HEIGHT)),
        (255, 255, 255),
        3,
    )

    # Score box
    cv.putText(
        frame,
        "Score",
        (margin, int(0.25 * CAM_HEIGHT)),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (200, 200, 200),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        str(human_score),
        (margin * 4, int(0.38 * CAM_HEIGHT)),  # TODO: Use true center alignment
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
        cv.LINE_AA,
    )
    cv.rectangle(
        frame,
        (margin, int(0.3 * CAM_HEIGHT)),
        (CAM_X_START - margin, int(0.4 * CAM_HEIGHT)),
        (200, 200, 200),
        2,
    )

    # Detected
    cv.putText(
        frame,
        "Detected",
        (margin, int(.5 * CAM_HEIGHT)),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (200, 200, 200),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        "Scissors",
        (margin * 2, int(0.63 * CAM_HEIGHT)),  # TODO: Use true center alignment
        cv.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv.LINE_AA,
    )
    cv.rectangle(
        frame,
        (margin, int(0.55 * CAM_HEIGHT)),
        (CAM_X_START - margin, int(0.65 * CAM_HEIGHT)),
        (200, 200, 200),
        2,
    )


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
