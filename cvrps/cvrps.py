from ui import *
from vision import *
import random
import common


def main() -> None:
    """
    Run program
    :return: None
    """

    # Global Variables
    game_state = GameState.START
    win_state = WinState.HUMAN
    human_score = 0
    computer_score = 0
    played_class = "..."

    # Extract arguments from CLI
    args = configure_argparse().parse_args()

    # Setup PyTorch
    configure_torch_gpu()
    load_model(args.model_path)

    # Initialize camera
    cam = initialize_camera(args.camera_idx)

    while cam.isOpened():
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Quit if frame is not read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Scale frame to target size
        # frame = scale_frame(frame)
        frame = scale_and_flip(frame)

        # Run based on state
        match game_state:
            case GameState.START:
                draw_start_screen(frame)
            case GameState.COUNT:
                # Draw countdown
                if draw_count_down(frame):
                    game_state = GameState.EVAL
            case GameState.EVAL:
                # Run the current frame through the model
                if not make_prediction(frame):
                    game_state = GameState.FAILED
                    continue

                # Computer pick a gesture
                played_class = random.choice(CLASSES[1:])

                # Determine winner
                if detected_class == played_class:
                    win_state = WinState.TIE
                elif (detected_class == "rock" and played_class == "paper") or (
                        detected_class == "paper" and played_class == "scissors") or (
                        detected_class == "scissors" and played_class == "rock"):
                    computer_score += 1
                    win_state = WinState.COMPUTER
                else:
                    human_score += 1
                    win_state = WinState.HUMAN

                game_state = GameState.RESULT
            case GameState.FAILED:
                if draw_failed_screen(frame):
                    game_state = GameState.START
            case GameState.RESULT:
                if draw_result_screen(frame, win_state):
                    game_state = GameState.START
            case _:
                pass

        # Draw UI
        draw_ui(frame, human_score, computer_score, played_class)

        # Display the full frame
        cv.imshow(WINDOW_NAME, frame)

        # Press Q on keyboard to exit
        if cv.waitKey(1) == ord("q"):
            break
        if cv.waitKey(1) == ord(" ") and game_state == GameState.START:
            game_state = GameState.COUNT

    # Cleanup
    print("Cleaning up")
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
