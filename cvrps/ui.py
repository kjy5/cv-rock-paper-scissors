from common import *
import cv2 as cv
import numpy as np

# Global variables
frame_needs_scaling = False


def scale_frame(frame: np.ndarray) -> np.ndarray:
    """
    Scale frame to target size
    Remembers if scaling is needed for next frame
    :param frame: Webcam frame to scale
    :return: Scaled frame
    """
    global frame_needs_scaling
    if frame_needs_scaling or frame.shape != CAM_SHAPE:
        out = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        frame_needs_scaling = True
        return out
    return frame


def put_text(
    frame: np.ndarray,
    text: any,
    location: tuple[any, any],
    size: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Put text centered on location
    :param frame: Webcam frame to draw on
    :param text: Text to draw
    :param location: Center of text
    :param size: Font size
    :param color: Color of text
    :param thickness: Thickness of text
    :return: None
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv.getTextSize(str(text), font, size, thickness)
    cv.putText(
        frame,
        str(text),
        (
            int(location[0]) - text_size[0] // 2,
            int(location[1]) - baseline // 8 + text_size[1] // 2,
        ),
        font,
        size,
        color,
        thickness,
        cv.LINE_AA,
    )


def scale_and_flip(frame: np.ndarray) -> np.ndarray:
    """
    Scale and flip frame
    :param fame: Webcam frame to scale and flip
    :return: Scaled and flipped frame
    """
    return cv.flip(scale_frame(frame), 1)


def draw_start_screen(frame: np.ndarray) -> None:
    """
    Draw start screen prompt
    :param frame: Webcam frame to draw on
    """
    start_text = "Press [SPACE] to play, [Q] to quit"
    # text_size, baseline = cv.getTextSize(str(text), font, size, thickness)
    # cv.rectangle(frame, (0, 0), (CAM_WIDTH, CAM_HEIGHT), (15, 15, 15), -1)
    put_text(
        frame,
        start_text,
        (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT),
        1,
        (255, 255, 255),
        3,
    )


def draw_ui(frame: np.ndarray) -> None:
    """
    Draw UI
    :param frame: Webcam frame to draw on
    :return: None
    """

    # Flip to help with visuals
    # cv.flip(frame, 1, frame)

    # Variables
    margin = int(0.02 * CAM_WIDTH)

    for i in range(2):
        # Create text values
        x_offset = 0
        left_coord = CAM_X_START
        title_text = "YOU"
        score_val = human_score
        detected_title_text = f"Detected ({detected_confidence}%)"
        detected_text = detected_class
        if i == 1:
            x_offset = CAM_X_END
            left_coord = CAM_WIDTH
            title_text = "COM."
            score_val = computer_score
            detected_title_text = "Played"
            detected_text = played_class

        # Background
        cv.rectangle(frame, (x_offset, 0), (left_coord, CAM_HEIGHT), (15, 15, 15), -1)

        # Title
        put_text(
            frame,
            title_text,
            (x_offset + CAM_X_START // 2, 0.1 * CAM_HEIGHT),
            2,
            (255, 255, 255),
            3,
        )
        cv.line(
            frame,
            (x_offset + margin, int(0.15 * CAM_HEIGHT)),
            (left_coord - margin, int(0.15 * CAM_HEIGHT)),
            (255, 255, 255),
            3,
        )

        # Score box
        put_text(
            frame,
            "Score",
            (x_offset + CAM_X_START // 2, 0.25 * CAM_HEIGHT),
            1,
            (200, 200, 200),
            2,
        )
        put_text(
            frame,
            score_val,
            (x_offset + CAM_X_START // 2, 0.35 * CAM_HEIGHT),
            2,
            (255, 255, 255),
            3,
        )
        cv.rectangle(
            frame,
            (x_offset + margin, int(0.3 * CAM_HEIGHT)),
            (left_coord - margin, int(0.4 * CAM_HEIGHT)),
            (200, 200, 200),
            2,
        )

        # Detected / Played
        put_text(
            frame,
            detected_title_text,
            (x_offset + CAM_X_START // 2, 0.5 * CAM_HEIGHT),
            1,
            (200, 200, 200),
            2,
        )
        put_text(
            frame,
            detected_text,
            (x_offset + CAM_X_START // 2, 0.6 * CAM_HEIGHT),
            1.5,
            (255, 255, 255),
            3,
        )
        cv.rectangle(
            frame,
            (x_offset + margin, int(0.55 * CAM_HEIGHT)),
            (left_coord - margin, int(0.65 * CAM_HEIGHT)),
            (200, 200, 200),
            2,
        )
