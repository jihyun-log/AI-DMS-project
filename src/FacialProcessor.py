from src.faceMesh import FaceMesh
from src.iris import Iris
import src.conf as conf
from src.eye import Eye

class FacialProcessor:
   def __init__(self):
      self.face_mesh = FaceMesh()
      self.detected = False
      self.eyes_status = "Eyes not detected"
      self.face_mesh_enabled = True  # FaceMesh 처리를 기본적으로 활성화

      # face mesh toggle 키 만들기
   def toggle_face_mesh(self):
      self.face_mesh_enabled = not self.face_mesh_enabled

   def process_frame(self, frame):
      if self.face_mesh_enabled:
      # FaceMesh 처리 코드
         self.face_mesh.process_frame(frame)

         # 원본 프레임에 대한 다른 처리 (여기에 코드 추가)
         self.face_mesh.frame = frame
         self.face_mesh.draw_mesh()

         # Add your logic to detect eyes and update self.detected and self.eyes_status
         # For demonstration purposes, let's assume eyes are always detected.
         self.detected = True
         self.eyes_status = "Eyes detected"

         if self.face_mesh.mesh_result.multi_face_landmarks:
            for face_landmarks in self.face_mesh.mesh_result.multi_face_landmarks:
               left_iris = Iris(frame, face_landmarks, conf.LEFT_EYE)
               right_iris = Iris(frame, face_landmarks, conf.RIGHT_EYE)
               left_iris.draw_iris(True)
               right_iris.draw_iris(True)

               # Add eye-tracking logic here
               left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
               right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
               left_eye.iris.draw_iris()
               right_eye.iris.draw_iris()

               if left_eye.eye_closed() or right_eye.eye_closed():
                  self.eyes_status = 'Eye closed'
               else:
                  if left_eye.gaze_right() and right_eye.gaze_right():
                     self.eyes_status = 'Gazing right'
                  elif left_eye.gaze_left() and right_eye.gaze_left():
                     self.eyes_status = 'Gazing left'
                  elif left_eye.gaze_center() and right_eye.gaze_center():
                     self.eyes_status = 'Gazing center'

         # Draw the face mesh after drawing the iris
         self.face_mesh.draw_mesh()