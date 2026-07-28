# Psychological Counseling Chatbot

An intelligent psychological counseling chatbot with:

- Emotion recognition using a webcam
- An integrated API to provide flexible responses
- Prompt engineering for customised responses in specific situations

![Chatbot Interface](chatbot.png)

## Pipeline

```text
Start face detection using YOLOv8
        ↓
Infer the user's emotion using the trained emotion recognition model
        ↓
Receive the user's text message
        ↓
Combine the emotion information and user message
        ↓
Send the combined input to the API with customised prompt settings
        ↓
Generate and display the response
```
![Chatbot pipeline](chatbot.png)

## Application Framework

```text
Application
└── Psychological_Counseling_Chatbot.py
    ├── API integration
    ├── User interface launch
    └── Emotion recognition
        ├── yolov8_face.py
        │   └── Face detection using the existing YOLOv8 model and parameters
        ├── inference.py
        │   └── Emotion inference using ConvNeXtV2-Pico trained on AffectNet
        └── emotion_buffer.py
            └── Saves real-time emotion history and extracts emotion information for chatbot input
```

## Emotion Recognition Training

```text
train_poly.py
dataset.py
```

- **Model:** ConvNeXtV2-Pico
- **Dataset:** AffectNet

## Notes

The YOLOv8 model files and the trained emotion-recognition checkpoint are not included because of their large file sizes.

Testing scripts, intermediate development scripts, and historical versions are also not included.

