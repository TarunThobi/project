import cv2
import face_recognition
from camera import Camera
from facedet import FaceDetector
from facerec import FaceRecognizer
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

cap = Camera()

face_detector = FaceDetector()

face_recognizer = FaceRecognizer(sfr.known_face_encodings, sfr.known_face_names)

while True:
    ret, frame = cap.read_frame()

    face_locations = face_detector.detect_faces(frame)

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = face_recognizer.recognize_faces(face_encodings)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cap.show_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
