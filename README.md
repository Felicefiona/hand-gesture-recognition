\# Real-Time Hand Gesture Recognition



A real-time computer vision project that recognizes hand gestures from a webcam using MediaPipe and a custom-trained machine learning model.



\## Gestures

Fist, Open Palm, Peace, Thumbs Up



\## How it works

1\. \*\*OpenCV\*\* captures live video from the webcam.

2\. \*\*MediaPipe Hands\*\* detects 21 landmarks on the hand.

3\. Landmark coordinates are normalized (relative to the wrist and scaled), so hand position and size don't matter.

4\. A \*\*Random Forest classifier\*\* (scikit-learn), trained on my own collected dataset, predicts the gesture.

5\. The predicted gesture and confidence are shown live on the video.



\## Results

\- Dataset: 1,339 samples collected by me (fist, palm, peace, thumbs up)

\- Test accuracy: 99.6% on a held-out 20% test set

\## Tech stack

Python, OpenCV, MediaPipe, scikit-learn, NumPy



\## Setup

```

py -3.12 -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



\## Run

```

python step3\_collect\_data.py   # collect your own data (keys 1-4)

python step4\_train\_model.py    # train the model

python step5\_live\_predict.py   # live recognition

```



\## Future improvements

\- Add more gestures

\- Control volume or slides with gestures

\- Two-hand support

