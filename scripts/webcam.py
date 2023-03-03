# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# https://pytorch.org/tutorials/intermediate/realtime_rpi.html
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
import torchvision

device = torch.device("cpu")
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

# Set up preprocessing for the image
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Train this model with the notebook
model = torch.load("model.pth")
model = torch.jit.script(model)
model.to(device)
model.eval()

with open("data/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_time = 0
with torch.no_grad():
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

        # Convert to RGB from BGR
        image = frame[:, :, [2, 1, 0]]
        input_tensor = preprocess(image).to(device)

        # Run it through model
        output = model(input_tensor.unsqueeze(0))
        probabilities = torch.softmax(output[0], dim=0)

        # Show top categories per image
        top_prob, top_catid = torch.topk(probabilities, 10)
        message = ""
        for i in range(top_prob.size(0)):
            message += f"{top_prob[i].item()*100:.2f}% {categories[top_catid[i]]}\n"
        print(message)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
