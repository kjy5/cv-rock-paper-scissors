from ui import *
from vision import *
import common

# Global Variables
game_state = GameState.START


def main() -> None:
    """
    Run program
    :return: None
    """

    # Extract arguments from CLI
    args = configure_argparse().parse_args()

    # Setup PyTorch
    configure_torch_gpu()
    load_model(args.model_path)

    # Initialize camera
    cam = initialize_camera(args.camera_idx)

    global game_state
    while cam.isOpened():
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Quit if frame is not read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Run the current frame through the model
        make_prediction(frame)

        # Scale frame to target size
        # frame = scale_frame(frame)
        frame = scale_and_flip(frame)

        # Run based on state
        match game_state:
            case GameState.START:
                draw_start_screen(frame)
            case GameState.COUNT:
                if draw_count_down(frame):
                    game_state = GameState.EVAL
            case GameState.EVAL:
                pass
            case _:
                pass

        # Draw UI
        draw_ui(frame)

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
