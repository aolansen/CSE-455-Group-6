import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import learner

#Model Loading
model = learner.model
model.load_state_dict(torch.load('emotion_detector_epoch_3.pth', map_location=torch.device('cpu')))
model.eval()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Copied from Preprocessor.py
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#Pressesing Face Emotion from OpenCV to match preprocessor.py
def preprocess(face):
    face_pil = Image.fromarray(face)
    face_transformed = data_transforms(face_pil)
    face_transformed = face_transformed.unsqueeze(0)
    return face_transformed

#Accessed our emotion
def get_emotion(face):
    with torch.no_grad():
        emotion = model(preprocess(face))
        _, prediction = torch.max(emotion, 1)
    return prediction

#Uses Machine Default Camera for input
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converts to Grayscale and then Use Model to Detech Face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draws the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #Crops face in frame
        face = gray[y:y+h, x:x+w]

        # Convert cropped face to RGB to go back into our model and gets the emotion
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        emotion_label = get_emotion(face_rgb)
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        emotion_text = emotion_names[emotion_label]

        # Put emotion on the Frame
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
