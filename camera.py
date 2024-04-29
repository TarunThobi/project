import cv2

class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def show_frame(self, frame):
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
