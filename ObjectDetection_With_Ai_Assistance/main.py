import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai

#NOTE:to exit the camera press "q"
#to start camera speak object detection
# you can normally access the ai by speaking with it after executing the code

# Function to speak text
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text 
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCBpdz0-cUbkJgbZx3972XojE3lBPA_NOY")
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Define conversation prompt
prompt = "1. If the question is technical, write answers in 2 to 3 lines which are understandable to us." \
         "2. If the question is a story, give a short story." \
         "3. If the question is creative, give creative answers." \
         "4. Give answers in paragraph format." \
         "5. If the question is a biography of a person, answer in short including all his events in short." \
         "6. Answer the questions in an efficient manner." \
         "7. Do not include asterisk in any of the answers."

# Start the conversation
while True:
    speech_text = recognize_speech()

    if speech_text:
        if "hello" in speech_text.lower():
            print("Hello! What can I do for you?")
            speak("Hello! What can I do for you?")
            convo = model.start_chat(history=[])
            while True:
                speech_text = recognize_speech()  # Listen to user's response
                if speech_text:
                    if speech_text.lower() == "exit":
                        print("Exiting conversation...")
                        speak("Goodbye! Have a nice day")
                        break
                    elif "object" in speech_text.lower():
                        print("Starting camera...")
                        speak("Starting camera...")
                        # Object detection loop
                        while True:
                            # Capture frame-by-frame
                            ret, frame = cap.read()
                            height, width, channels = frame.shape

                            # Detect objects
                            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                            net.setInput(blob)
                            outs = net.forward(output_layers)

                            # Process detection results
                            for out in outs:
                                for detection in out:
                                    scores = detection[5:]
                                    class_id = np.argmax(scores)
                                    confidence = scores[class_id]
                                    if confidence > 0.5:
                                        # Object detected
                                        center_x = int(detection[0] * width)
                                        center_y = int(detection[1] * height)
                                        w = int(detection[2] * width)
                                        h = int(detection[3] * height)

                                        # Rectangle coordinates
                                        x = int(center_x - w / 2)
                                        y = int(center_y - h / 2)

                                        # Draw rectangle and label
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Display the resulting frame
                            cv2.imshow('Object Detection', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        # Release the capture
                        cap.release()
                        cv2.destroyAllWindows()

                        print("Searching the recognized object in Gem AI...")
                        speak("Searching the recognized object in Gem AI...")
                        # Here, you can add code to search the recognized object in Gem AI and get 2 sentences about it.

                        break
                    else:
                        # Concatenate user's input with the requirement prompt
                        prompt = prompt + " " + speech_text
                        convo.send_message(prompt)
                        response = convo.last.text
                        print("Response:", response)
                        speak(response)
                else:
                    print("No speech detected.")
            break
        elif speech_text.lower() == "exit":
            print("Exiting conversation...")
            speak("Goodbye!")
            break
        else:
            print("Please say 'hello' to start the conversation.")
    else:
        print("No speech detected.")
