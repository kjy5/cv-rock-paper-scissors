from ui import *
from vision import *


def main() -> None:
    """
    Run program
    :return: None
    """

    # Extract arguments from CLI
    args = configure_argparse().parse_args()

    # Setup PyTorch
    configure_torch_gpu()

    # Initialize camera
    cam = initialize_camera(args.camera_idx)

    while cam.isOpened():
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Quit if frame is not read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Save a copy of the frame to process
        to_process_frame = np.copy(frame)

        # Scale frame to target size
        frame = scale_frame(frame)

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
