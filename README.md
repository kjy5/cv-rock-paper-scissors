# Computer Vision Rock Paper Scissors

CSE 455 Wi23 final project by [David Stumph](https://github.com/Davester47)
and [Kenneth Yang](https://github.com/kjy5).

![Demo](demo.gif)

We created a program that would allow a human to play Rock Paper Scissors with a
computer, without needing to use a keyboard or mouse to enter in their move.

[Video Presentation](https://youtu.be/FDJvC4X2EH0)

# Installation and Usage

## System requirements

- Python 3.10+
- Some OpenCV compatible camera or webcam
- Recommended: a CUDA-enabled GPU or a Apple Silicon Mac (MPS backends)
    - We did not test against AMD GPUs.

## Installation

1. Download our pre-trained `model.pth`
   from [here](https://drive.google.com/file/d/13Pdq7E35vu-gj2wpS6unCe_6bfIe9G4K/view?usp=share_link).
    1. Alternatively, you can train your own model by running
       through `scripts/train.ipynb`.
    2. Or use
       Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjy5/cv-rock-paper-scissors/blob/main/scripts/train.ipynb)
2. Clone the repository.
3. Install the dependencies with `pip install -r requirements.txt`.
4. Run `python -m cvrps -m [location of model.pth]` to start the game.
    1. If there was an error finding the webcam, use the `-c` flag to specify
       the camera index.

# Writeup

## Data Collection

For the training data, we took pictures and videos of ourselves and a couple
friends making the gestures for rock, paper, and scissors at the camera. We
shortened and unrolled each video into frames, and added them to the training
set. We also used a subset of about 1000 photos from
the [Tiny ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
to train as a “clutter” class, so our model would be able to detect “no gesture”
rather than just guess a gesture if the image had none.

## Techniques

To gather data used for training our model, we used a greenscreen background and
a camera with a high shutter speed. This allowed us to replace the background
with random “clutter” images from the Tiny ImageNet dataset to increase the
generalizability of our data and to also reduce motion blur in our images. The
video also allowed us to capture the hand gestures at various angles which will
be more specific than using a existing data model on similar hand gestures like
an ASL data set.

Tiny ImageNet contains 10,000 photos. To ensure we didn’t have a
disproportionate amount of clutter images compared to our training images we
decided to reduce the number of photos down to roughly 1000 by taking every
image that had a number starting with 2 as our subset, because otherwise there
would have been way more clutter images than RPS images.

For our model, we used PyTorch and started with a version of ResNet-34
pre-trained on ImageNet 1000. We then replaced the final fully-connected layer
of the model with one that had 4 output classes instead of 1000, and we
retrained the model on our dataset. We experimented with only training the final
layer, but we found that it did not work nearly as well and so we fine-tuned the
full model.

For our training parameters, we used a batch size of 16, a training/validation
split of 0.8/0.2, cross-entropy loss, and an initial learning rate of 0.0001
with Adam as the optimizer. After 20 epochs, we decreased the learning rate by a
factor of 10 and trained for 10 more epochs. We also experimented with deeper
versions of ResNet but found that they didn’t perform noticeably better in
real-time and were considerably more expensive to run.

## Resources and Tutorials used

We made use of pre-trained ResNet included in PyTorch as our model, as well as
some tutorials and references on how to use it. Those references include the
section notebooks from this quarter’s 446 Machine Learning class, as well as
PyTorch’s [tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Stuff we made from scratch

We created our own training dataset by taking videos and pictures of ourselves,
and we also wrote the training code for the model (with some inspiration from
the sources in the previous section). We created our own custom UI with OpenCV
for the game itself.

## Ideas for future work

We had a difficult time getting our model to generalize well, and gathering a
much larger dataset would have helped. Ideally, we would collect the gestures
from a lot more than just three people, with a wide range of skin colors
included.

Improvements to the UI could also be made, but that was not the focus of the
project.
In particular, the prompting mechanism is a little clumsy. In the future, we
could
have also trained the model to recognize when a gesture has been made, so the
user
wouldn't need to wait on the prompt and they could just make a move.

See our code [on GitHub](https://github.com/kjy5/cv-rock-paper-scissors)

[![pages-build-deployment](https://github.com/kjy5/cv-rock-paper-scissors/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/kjy5/cv-rock-paper-scissors/actions/workflows/pages/pages-build-deployment)