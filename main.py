import torch
import os
import cv2
import dlib
import time
import pygame

import tensorflow as tf
import psutil
import imutils

import datetime
import collections
import numpy as np
import src.conf as conf

from imutils import face_utils
from scipy.spatial import distance
from src.FacialProcessor import FacialProcessor

# == GPU pytorch =======================================================================================================
tf.config.list_physical_devices('GPU')
print()

# torch의 백엔드를 mps로 설정

print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")
device = torch.device("mps")

# =======================================================================================================================

# == init ==============================================================================================================
detect = dlib.get_frontal_face_detector()
path = './haar/'
predict = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')
pygame.mixer.init()
sound_siren = pygame.mixer.Sound("/Users/bagsangbeom/PycharmProjects/DMS_Project/sound/siren.wav")
sound_drowsy_warning = pygame.mixer.Sound("/Users/bagsangbeom/PycharmProjects/DMS_Project/sound/drowsy_warning.wav")
sound_drowsy_detection = pygame.mixer.Sound("/Users/bagsangbeom/PycharmProjects/DMS_Project/sound/drowsy_detection.wav")
facial_processor = FacialProcessor()
# =======================================================================================================================

# == capture ===========================================================================================================
# 1. function
# 이벤트 발생시 캡처화면 생성
def save_frame(frame):
	# flipped_frame = cv2.flip(frame, 1)  # 좌우 반전
	# 현재 시각을 기반으로 파일 이름을 생성한다.
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"capture_{timestamp}.jpg"
	filepath = os.path.join("./save/sleep/capture", filename)  # 저장할 경로 지정
	cv2.imwrite(filepath, frame)
	return filepath


# 이벤트 발생 시 영상 저장을 시작
def start_saving_event_video(frame_width, frame_height):
	# 현재 시각을 기반으로 파일 이름을 생성
	global event_occurred, event_frame_count
	event_occurred = True
	event_frame_count = 0
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"event_{timestamp}.mp4"
	# 저장할 경로 설정
	save_path = os.path.join("./save/sleep/videos", filename)
	video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
	while frames_buffer:
		buffer_frame = cv2.flip(frames_buffer.popleft(), 1)  # 버퍼 프레임 반전
		video_writer.write(buffer_frame)
	return video_writer


# 2. parameter
# 동영상 저장을 위한 프레임 설정
buffer_size = 2  # 버퍼 크기 2초
fps = 30  # 초당 프레임 수
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 비디오 코덱 설정

# 프레임 버퍼 초기화
frames_buffer = collections.deque(maxlen=buffer_size * fps)  # 프레임을 저장할 수

# 이벤트 발생 시 영상을 저장할 준비
event_occurred = False
event_frame_count = 0  # 이벤트 발생 후 프레임 수

# 이벤트 후 녹화할 추가 시간 설정(5초)
extra_recording_time = 5 * fps  # 5초( * fps는 프레임 단위로 변경)
post_event_frames = buffer_size * fps  # 이벤트 후 저장할 프레임 수

# 캡처 및 영상 녹화 ON/OFF
capture_enabled = True
recording_enabled = True

# 캡처 및 영상 녹화 상태 출력
capture_status_text = "Capture ON"
recording_status_text = "Record ON"


# =======================================================================================================================


# == YOLO ==============================================================================================================
# #Load a model
# model = YOLO('/Users/bagsangbeom/PycharmProjects/DMS/weights/best (12월 16일).pt')  # build a new model from YAML
#
# # confidence 임계치
# DETECT_CONFIDENCE_THRESHOLD = 0.65
# BOX_CONFIDENCE_THRESHOLD = 0.4
#
# # 각 클래스에 대한 카운트를 저장하는 변수
# calling_count = 0
# smoking_count = 0
# texting_count = 0
#
# # 각 클래스에 대한 limit 저장하는 변수
# CALLING_LIMIT = 3
# SMOKING_LIMIT = 3
# TEXTING_LIMIT = 3
#
# # 감지에 대한 알림 함수
# def send_notification(class_name):
#     # 여기에 실제 알림 로직을 구현하세요.
#     print(f"{class_name} 감지에 대한 알림을 보냅니다.")
# =======================================================================================================================

# == Drowsiness Detection ==============================================================================================
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
leftEAR = 0.0
rightEAR = 0.0

ptime = 0
ctime = 0
number_closed = 0

# 졸음 감지를 위한 변수 설정
facial_processor = FacialProcessor()
thresh = 0.2
warn_limit = 10  # -- 1차 알림 FPS 7~10
detect_limit = 60  # -- 2차 알림 FPS 7~10
previous_warn_state = False  # -- 이전 경고가 울린 상태인지 여부를 저장하는 변수

# 졸음 감지 후 지속적인 알람을 위한 추가 변수 설정
continuous_alarm_count = 0
alarm_interval = 30  # 3차 알람 간격 (프레임 수 기준) FPS 7~10
# =======================================================================================================================


# == webcam ============================================================================================================
# Adjust the capture device index (0 or 1) based on your camera setup
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height of the frames
cap.set(cv2.CAP_PROP_FPS, 10)  # 프레임 설정

while True:
	ret, frame = cap.read()
	# ret이 false일 경우(카메라에 문제가 생겨 꺼졌거나 프레임 반응이 없을때) 무한루프에 돌지 않도록 종료(break)시켜주는 코드
	if not ret:
		break
	
	# CPU 사용률 가져오기
	cpu_usage = psutil.cpu_percent()
	
	# flipped_frame = cv2.flip(frame, 1)
	#
	# frames_buffer.append(flipped_frame)  # 프레임 버퍼에 뒤집힌 프레임 추가
	
	# Check if frame is not None before processing
	if frame is not None:
		frame = imutils.resize(frame, width=1640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Use the face detector
		subjects = detect(gray)
		
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)
			
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			
			if ear > thresh:
				number_closed -= 1
				print(number_closed)
				if (number_closed < 0):
					number_closed = 0
			
			elif ear < thresh:
				number_closed += 1
				print(number_closed)
				
				if number_closed == warn_limit and not previous_warn_state:
					sound_drowsy_warning.play()
					print("Warning : Drowsiness")
					previous_warn_state = True
				
				elif number_closed == detect_limit:
					cv2.putText(frame, "", (10, 30),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					
					# 경고 알람 재생
					sound_drowsy_detection.play()
					print("Drowsiness Detected")
					
					# 캡처 진행
					if capture_enabled:
						capture_status_text = "Capture ON" if capture_enabled else "Capture OFF"
						# 캡처하고 파일 경로를 반환받는다.
						capture_path = save_frame(frame)
						print(f"Capture image saved to: {capture_path}")
					else:
						capture_status_text = "Capture OFF"
					
					# 영상 이벤트 진행
					if recording_enabled:
						recording_status_text = "Record ON" if recording_enabled else "Record OFF"
						# 이벤트 발생 시점이면, 동영상 저장 준비
						if not event_occurred:
							frame_width = frame.shape[1]
							frame_height = frame.shape[0]
							video_path = start_saving_event_video(frame_width, frame_height)
							# 버퍼에 저장된 프레임(이벤트 발생 전 2초)을 영상 파일에 먼저 기록
							while frames_buffer:
								video_path.write(frames_buffer.popleft())
							event_occurred = True
							event_frame_count = 0  # 프레임 카운트 초기화
					else:
						recording_status_text = "Record OFF"
				
				# 이벤트가 지속되는 동안 프레임을 기록
				if event_occurred and event_frame_count < post_event_frames:
					video_path.write(frame)
					event_frame_count += 1
				
				# 이벤트 후 추가 시간(5초)까지 녹화 완료
				elif event_occurred and event_frame_count >= post_event_frames:
					video_path.release()
					event_occurred = False
					frames_buffer.clear()
				
				# 사용자가 눈을 감고 있어도 지속적으로 알람을 주기
				if number_closed > detect_limit:
					# 졸음 상태에 있을때 카운트를 1씩 증가
					continuous_alarm_count += 1
					
					# 졸음 카운트가 설정된 알람 간격에 도달했는지 체크 위에 30으로 설정
					if continuous_alarm_count >= alarm_interval:
						sound_siren.play()
						print("Continuous Drowsiness Detected")
						# 알람을 울리면 카운트를 다시 0으로 초기화
						continuous_alarm_count = 0
			
			else:
				if event_occurred:
					if event_frame_count < (buffer_size * fps + extra_recording_time):
						# 이벤트 후 추가 시간 동안 계속 녹화
						video_path.write(frame)
						event_frame_count += 1
					else:
						# 추가 시간이 끝나면 녹화 종료
						video_path.release()
						event_occurred = False
						frames_buffer.clear()
				if event_occurred and event_frame_count >= post_event_frames:
					video_path.release()
					event_occurred = False
					frames_buffer.clear()
					
					# 데이터베이스에 파일 정보 저장
					# user_id = 1  # 예시 사용자 ID, 실제 사용자 ID로 교체 필요
					# save_file_info_to_db(user_id, capture_path, video_path)
				
				# 버퍼를 다시 초기화하여 다음 이벤트 대비
				frames_buffer.clear()
			
			sign = 'sleep count : ' + str(number_closed) + ' / ' + str(warn_limit)
		
		# FaceMesh_on_off
		if facial_processor.face_mesh_enabled:
			# FaceMesh 프로세싱 및 원본 프레임에 그리기
			facial_processor.process_frame(frame)
			
			# 새로운 프레임 생성 및 검은색으로 초기화
			face_mesh_frame = np.zeros_like(frame)
			
			# 원본 프레임에 대한 처리
			facial_processor.face_mesh.frame = frame
			facial_processor.face_mesh.draw_mesh()
			facial_processor.face_mesh.draw_mesh_eyes()
			
			# FaceMesh Frame에 추가 정보 표시
			facial_processor.face_mesh.frame = face_mesh_frame
			facial_processor.face_mesh.draw_mesh()
		
		# fps 측정
		ctime = time.time()
		fps = 1 / (ctime - ptime)
		ptime = ctime
		
		shadow_offset = 2  # 그림자 오프셋 값
		
		# YOLO Load
		# results = model(frame, stream=True)
		# frame = yolo_pred(frame, results,
		#                   DETECT_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD,
		#                   CALLING_LIMIT, SMOKING_LIMIT, TEXTING_LIMIT)
		
		# == cv2.putText 설정 ===========================================================================================
		# 화면에 fps 표시
		cv2.putText(face_mesh_frame, f'FPS: {int(fps)}', (10 + shadow_offset, 30 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, f'FPS: {int(fps)}', (10, 30),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		# 화면에 CPU 사용률 표시
		cv2.putText(face_mesh_frame, f'CPU Usage: {cpu_usage}%', (10 + shadow_offset, 60 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, f'CPU Usage: {cpu_usage}%', (10, 60),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		
		# 눈 : 상태 화면에 표시
		cv2.putText(face_mesh_frame, f'{facial_processor.eyes_status}', (10 + shadow_offset, 800 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, f'{facial_processor.eyes_status}', (10, 800),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		
		# 눈 : 종횡비 화면에 표시
		cv2.putText(face_mesh_frame, "Left EAR {:.2f}".format(leftEAR), (10 + shadow_offset, 870 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, "Left EAR {:.2f}".format(leftEAR), (10, 870),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, "Right EAR {:.2f}".format(rightEAR), (10 + shadow_offset, 900 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
		cv2.putText(face_mesh_frame, "Right EAR {:.2f}".format(rightEAR), (10, 900),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
		
		# # 눈 : 감은 상태 화면에 표시
		# cv2.putText(face_mesh_frame, sign, (10, 330),
		#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf.TEXT_COLOR, 1)
		
		# FaceMesh ON/OFF 설정
		frame_width = frame.shape[1]
		text = "FaceMesh ON" if facial_processor.face_mesh_enabled else "FaceMesh OFF"
		text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
		text_x = frame_width - text_size[0]  # 오른쪽 여백 10 픽셀 고려
		text_y = 30  # 상단 여백
		
		# # FaceMesh 상태 화면에 텍스트로 표시
		cv2.putText(face_mesh_frame, text, (1440 + shadow_offset, text_y + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
		cv2.putText(face_mesh_frame, text, (1440, text_y),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2)
		
		# capture, recoding 상태 화면에 텍스트 표시
		cv2.putText(face_mesh_frame, capture_status_text, (1470 + shadow_offset, text_y + 30 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # 검은색 그림자
		# 원래 텍스트 (원래의 색상과 크기로)
		cv2.putText(face_mesh_frame, capture_status_text, (1470, text_y + 30),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2)
		
		cv2.putText(face_mesh_frame, recording_status_text, (1480 + shadow_offset, text_y + 60 + shadow_offset),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
		cv2.putText(face_mesh_frame, recording_status_text, (1480, text_y + 60),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf.TEXT_COLOR, 2)
		# 전체 통합 출력
		cv2.imshow('Original Frame', frame)
		cv2.imshow('FaceMesh Frame', face_mesh_frame)
	
	# Delay to control the frames per second
	key = cv2.waitKey(1) & 0xFF
	if key == ord('m') or key == ord('M'):
		facial_processor.toggle_face_mesh()
	if key == ord('c'):
		capture_enabled = not capture_enabled
		capture_status_text = "Capture ON" if capture_enabled else "Capture OFF"
	if key == ord('r'):
		recording_enabled = not recording_enabled
		recording_status_text = "Record ON" if recording_enabled else "Record OFF"
	if key == ord('q'):
		break

# Release the capture and close all windows
cv2.destroyAllWindows()
cap.release()
