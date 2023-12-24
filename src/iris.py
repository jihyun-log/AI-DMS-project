import cv2
import time
import src.conf as conf
from src.faceMesh import FaceMesh
import numpy as np

class Iris:

    def __init__(self, frame, face_landmarks, id, is_left=True):  # is_left 매개변수 추가
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id
        self.is_left = is_left  # is_left 속성 추가
        self.pos = self._get_iris_pos()

    def _get_iris_pos(self):
        h, w = self.frame.shape[:2]
        iris_pos = list()
        for id in self.id[-5:]:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            iris_pos.append((cx, cy))
        return iris_pos

    def draw_iris(self, border=False):
        cv2.circle(self.frame, self.pos[0], 2, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)

        if border:
            for pos in self.pos[1:]:
                cv2.circle(self.frame, pos, 1, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)

        if len(self.pos) == 2:
            center_left = ((self.pos[0][0] + self.pos[1][0]) // 2, (self.pos[0][1] + self.pos[1][1]) // 2)
            center_right = center_left
            cv2.line(self.frame, center_left, center_right, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            arrow_size = 20
            angle = int(180 / 3.14 * (self._get_angle(self.pos[0], self.pos[1]) + 3.14))
            p1 = (center_right[0] + int(arrow_size * np.cos(angle + 3.14 / 6)),
                  center_right[1] + int(arrow_size * np.sin(angle + 3.14 / 6)))
            p2 = (center_right[0] + int(arrow_size * np.cos(angle - 3.14 / 6)),
                  center_right[1] + int(arrow_size * np.sin(angle - 3.14 / 6)))
            cv2.line(self.frame, center_right, p1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(self.frame, center_right, p2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(self.frame, p1, center_right, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(self.frame, p2, center_right, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    def _get_angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

class CombinedIrisTracker:

    def __init__(self):
        self.cap = cv2.VideoCapture(conf.CAM_ID)
        self.cap.set(3, conf.FRAME_W)
        self.cap.set(4, conf.FRAME_H)
        self.fm = FaceMesh()
        self.ptime = 0
        self.ctime = 0

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            self.fm.process_frame(frame)
            if self.fm.mesh_result.multi_face_landmarks:
                for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                    left_iris = Iris(frame, face_landmarks, conf.LEFT_EYE, is_left=True)
                    right_iris = Iris(frame, face_landmarks, conf.RIGHT_EYE, is_left=False)
                    left_iris.draw_iris(True)
                    right_iris.draw_iris(True)

            self.ctime = time.time()
            fps = 1 / (self.ctime - self.ptime)
            self.ptime = self.ctime

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.8,
                        conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

            cv2.imshow('Iris tracking', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    iris_tracker = CombinedIrisTracker()
    iris_tracker.run()
