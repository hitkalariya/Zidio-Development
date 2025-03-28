import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but critical TensorFlow logs

import cv2
from deepface import DeepFace
import speech_recognition as sr
from transformers import pipeline, logging

# Suppress Hugging Face logs
logging.set_verbosity_error()  # Suppress unnecessary model initialization logs

# Text Emotion Detection
def text_emotion_detection():
    print("Text Emotion Detection:")
    while True:
        text = input("Enter text (or type 'back' to return to the menu): ")
        if text.lower() == "back":
            return
        try:
            emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
            result = emotion_classifier(text)
            print("Detected Emotion:", result[0]['label'])
        except Exception as e:
            print(f"Error during text emotion detection: {e}")

# Video Emotion Detection (Camera)
def video_emotion_detection():
    print("Video Emotion Detection:")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and len(analysis) > 0:
                dominant_emotion = analysis[0].get('dominant_emotion', 'No face detected')
            else:
                dominant_emotion = analysis.get('dominant_emotion', 'No face detected')
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            cv2.putText(frame, "Error detecting emotion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video Emotion Detection", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # Quit current mode
            print("Exiting Video Emotion Detection...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Speech Emotion Detection
def speech_emotion_detection():
    print("Speech Emotion Detection:")
    recognizer = sr.Recognizer()
    while True:
        print("\nSpeak into the microphone or type 'back' to return to the menu:")
        user_input = input("(Press Enter to start speaking or type 'back'): ")
        if user_input.lower() == "back":
            return
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print("You said:", text)

                # Analyze text emotions
                emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
                result = emotion_classifier(text)
                print("Detected Emotion:", result[0]['label'])
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
        except Exception as e:
            print(f"Error during speech emotion detection: {e}")

# Main Menu
def main():
    while True:
        print("\nChoose Emotion Detection Type:")
        print("1. Text Emotion Detection")
        print("2. Video Emotion Detection (via Camera)")
        print("3. Speech Emotion Detection (via Microphone)")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            text_emotion_detection()
        elif choice == '2':
            video_emotion_detection()
        elif choice == '3':
            speech_emotion_detection()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
