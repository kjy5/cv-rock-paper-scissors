from common import *
import vision
import cv2 as cv
import numpy as np
import time

# Global variables
frame_needs_scaling = False
count_start_time: float = 0


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


def put_text_rect(
        frame: np.ndarray,
        text: any,
        location: tuple[any, any],
        size: int,
        color: tuple[int, int, int],
        thickness: int,
) -> None:
    """
    Put rect based on text centered on location
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
    start_point = (
        int(location[0]) - text_size[0] // 2,
        int(location[1]) + baseline + text_size[1] // 2,
    )
    cv.rectangle(
        frame,
        start_point,
        (start_point[0] + text_size[0], start_point[1] - text_size[1] * 2),
        color,
        -1,
    )


def scale_and_flip(frame: np.ndarray) -> np.ndarray:
    """
    Scale and flip frame
    :param frame: Webcam frame to scale and flip
    :return: Scaled and flipped frame
    """
    global frame_needs_scaling
    scaled_frame = frame
    if frame_needs_scaling or frame.shape != CAM_SHAPE:
        scaled_frame = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        frame_needs_scaling = True
    return cv.flip(scaled_frame, 1)


def draw_start_screen(frame: np.ndarray) -> None:
    """
    Draw start screen prompt
    :param frame: Webcam frame to draw on
    """
    start_text = "Hold [SPACE] to start, [Q] to quit"
    put_text_rect(
        frame, start_text, (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT), 1, (100, 100, 100), 3
    )
    put_text(
        frame,
        start_text,
        (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT),
        1,
        (255, 255, 255),
        3,
    )


def draw_count_down(frame: np.ndarray) -> bool:
    """
    Draw count down
    :param frame: Webcam frame to draw on
    """
    global count_start_time
    if count_start_time == 0:
        count_start_time = time.time()
    count = int(5 - (time.time() - count_start_time))
    text: str
    match count:
        case 1:
            text = "SHOOT!!!"
        case 2:
            text = "SCISSORS"
        case 3:
            text = "PAPER"
        case 4:
            text = "ROCK"
        case _:
            count_start_time = 0
            return True
    put_text_rect(
        frame, text, (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT), 2, (100, 100, 100), 3
    )
    put_text(
        frame,
        text,
        (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT),
        2,
        (255, 255, 255),
        3,
    )
    return False


def draw_failed_screen(frame: np.ndarray) -> bool:
    """
    Draw failed screen prompt
    :param frame: Webcam frame to draw on
    """
    global count_start_time
    if count_start_time == 0:
        count_start_time = time.time()
    text = "No gesture detected, please try again"
    count = int(4 - (time.time() - count_start_time))
    put_text_rect(
        frame, text, (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT), 1, (100, 100, 100), 2
    )
    put_text(
        frame,
        text,
        (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT),
        1,
        (255, 255, 255),
        2,
    )

    if count == 0:
        count_start_time = 0
        return True
    return False


def draw_result_screen(frame: np.ndarray, win_state: WinState) -> bool:
    """
    Draw result screen
    :param frame: Webcam frame to draw on
    :param win_state: Win state
    """
    global count_start_time
    if count_start_time == 0:
        count_start_time = time.time()
    text = "You win!"
    color = (0, 255, 0)
    match win_state:
        case WinState.COMPUTER:
            text = "You lose!"
            color = (0, 0, 255)
        case WinState.TIE:
            text = "Tie!"
            color = (255, 255, 255)

    count = int(3 - (time.time() - count_start_time))
    put_text_rect(
        frame, text, (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT), 2, (100, 100, 100), 2
    )
    put_text(
        frame,
        text,
        (CAM_WIDTH // 2, 0.2 * CAM_HEIGHT),
        2,
        color,
        2,
    )
    if count == 0:
        count_start_time = 0
        return True
    return False


def draw_ui(frame: np.ndarray, human_score: int, computer_score: int,
            played_class: str) -> None:
    """
    Draw UI
    :param frame: Webcam frame to draw on
    :param human_score: Human score
    :param computer_score: Computer score
    :param played_class: Played class
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
        detected_title_text = f"Detected ({vision.detected_confidence}%)"
        detected_text = vision.detected_class.capitalize()
        if i == 1:
            x_offset = CAM_X_END
            left_coord = CAM_WIDTH
            title_text = "COM."
            score_val = computer_score
            detected_title_text = "Played"
            detected_text = played_class.capitalize()

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
