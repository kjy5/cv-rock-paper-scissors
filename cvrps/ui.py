import cv2 as cv
import numpy as np
from common import *


def draw_ui(frame: np.ndarray) -> None:
    """
    Draw UI
    :param frame: Webcam frame to draw on
    :return: None
    """
    # Resize to 720p
    frame = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

    # Flip to help with visuals
    cv.flip(frame, 1, frame)

    # Variables
    margin = int(0.02 * CAM_WIDTH)

    for i in range(2):
        # Create text values
        x_offset = 0
        left_coord = CAM_X_START
        title_text = "YOU"
        score_val = human_score
        detected_title_text = "Detected"
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
        cv.putText(
            frame,
            title_text,
            (x_offset + margin, int(.1 * CAM_HEIGHT)),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )
        cv.line(
            frame,
            (x_offset + margin, int(0.15 * CAM_HEIGHT)),
            (left_coord - margin, int(0.15 * CAM_HEIGHT)),
            (255, 255, 255),
            3,
        )

        # Score box
        cv.putText(
            frame,
            "Score",
            (x_offset + margin, int(0.25 * CAM_HEIGHT)),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            str(score_val),
            (x_offset + margin * 4, int(0.38 * CAM_HEIGHT)),
            # TODO: Use true center alignment
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )
        cv.rectangle(
            frame,
            (x_offset + margin, int(0.3 * CAM_HEIGHT)),
            (left_coord - margin, int(0.4 * CAM_HEIGHT)),
            (200, 200, 200),
            2,
        )

        # Detected / Played
        cv.putText(
            frame,
            detected_title_text,
            (x_offset + margin, int(.5 * CAM_HEIGHT)),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 200, 200),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            detected_text,
            (x_offset + margin, int(0.63 * CAM_HEIGHT)),
            # TODO: Use true center alignment
            cv.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )
        cv.rectangle(
            frame,
            (x_offset + margin, int(0.55 * CAM_HEIGHT)),
            (left_coord - margin, int(0.65 * CAM_HEIGHT)),
            (200, 200, 200),
            2,
        )
