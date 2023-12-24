import os
import cv2
import pygame
import datetime

pygame.mixer.init()
sound_siren = pygame.mixer.Sound("/Users/jihyun/PycharmProjects/AI-DMS-project/sound/siren.wav")
sound_calling = pygame.mixer.Sound("/Users/jihyun/PycharmProjects/AI-DMS-project/sound/calling.wav")
sound_smoking = pygame.mixer.Sound("/Users/jihyun/PycharmProjects/AI-DMS-project/sound/smoking.wav")
sound_texting = pygame.mixer.Sound("/Users/jihyun/PycharmProjects/AI-DMS-project/sound/texting.wav")

# object classes
classNames = ["calling", "smoking", "texting"]

# 각 클래스에 대한 카운트를 저장하는 변수
calling_count = 0
smoking_count = 0
texting_count = 0

# 이미지 캡처 함수
def capture_event_image(frame, event_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{event_name}_{timestamp}.jpg"
    # 이벤트 이름에 따라 저장 폴더 변경
    save_path = f"./save/{event_name}/capture"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = os.path.join(save_path, filename)
    cv2.imwrite(filepath, frame)
    print(f"Captured {event_name} image saved to: {filepath}")
    return filepath

def yolo_pred(frame, results,
              detect_confidence_threshold=0.65, box_confidence_threshold=0.4,
              calling_limit=3, smoking_limit=3, texting_limit=3):
    global calling_count
    global smoking_count
    global texting_count

    for r in results:
        boxes = r.boxes

        for box in boxes:

            confidence = box.conf[0] # 신뢰도
            cls = int(box.cls[0]) # class name
            x1, y1, x2, y2 = box.xyxy[0] # 경계 상자
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 정수 값으로 변환

            # 객체 세부 정보
            org = [x1, y1 - 10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 255, 255)
            thickness = 2
            text = f"{classNames[cls]} : {confidence:.2f}"

# == calling ===========================================================================================================
            # 처음 감지될 때
            if (calling_count == 0) and (confidence > detect_confidence_threshold):
                # calling 클래스인 경우
                if classNames[cls] == "calling":
                    # calling 카운트 업데이트
                    calling_count += 1
                    print("calling count : ", calling_count)
                    print("calling confidence : ", f"{confidence:.2f}")

                    if calling_count == calling_limit:
                        send_notification("calling")
                        sound_calling.play()

            # 감지가 시작된 후에는 confidence가 0.4 이상일 때에만 계속해서 감지
            elif (calling_count > 0) and (confidence > 0.4):
                # calling 클래스인 경우
                if classNames[cls] == "calling":
                    # calling 카운트 업데이트
                    calling_count += 1
                    print("calling count : ", calling_count)
                    print("calling confidence : ", f"{confidence:.2f}")

                    if calling_count == calling_limit:
                        send_notification("calling")
                        sound_calling.play()
            else:
                # confidence가 0.4 미만일 때는 감지 종료
                calling_count = 0


# == smoking ===========================================================================================================
            # 처음 감지될 때
            if (smoking_count == 0) and (confidence > detect_confidence_threshold):
                # smoking 클래스인 경우
                if classNames[cls] == "smoking":
                    # smoking 카운트 업데이트
                    smoking_count += 1
                    print("smoking count : ", smoking_count)
                    print("smoking confidence : ", f"{confidence:.2f}")

                    if smoking_count == smoking_limit:
                        send_notification("smoking")
                        sound_smoking.play()

            # 감지가 시작된 후에는 confidence가 0.4 이상일 때에만 계속해서 감지
            elif (smoking_count > 0) and (confidence > 0.4):
                # smoking 클래스인 경우
                if classNames[cls] == "smoking":
                    # smoking 카운트 업데이트
                    smoking_count += 1
                    print("smoking count : ", smoking_count)
                    print("smoking confidence : ", f"{confidence:.2f}")

                    if smoking_count == smoking_limit:
                        send_notification("smoking")
                        sound_smoking.play()
            else:
                # confidence가 0.4 미만일 때는 감지 종료
                smoking_count = 0

# == texting ===========================================================================================================
            # 처음 감지될 때
            if (texting_count == 0) and (confidence > detect_confidence_threshold):
                # texting 클래스인 경우
                if classNames[cls] == "texting":
                    # texting 카운트 업데이트
                    texting_count += 1
                    print("texting count : ", texting_count)
                    print("texting confidence : ", f"{confidence:.2f}")

                    if texting_count == texting_limit:
                        send_notification("texting")
                        sound_texting.play()

            # 감지가 시작된 후에는 confidence가 0.4 이상일 때에만 계속해서 감지
            elif (texting_count > 0) and (confidence > 0.4):
                # texting 클래스인 경우
                if classNames[cls] == "texting":
                    # texting 카운트 업데이트
                    texting_count += 1
                    print("texting count : ", texting_count)
                    print("texting confidence : ", f"{confidence:.2f}")

                    if texting_count == texting_limit:
                        send_notification("texting")
                        sound_texting.play()
            else:
                # confidence가 0.4 미만일 때는 감지 종료
                texting_count = 0




            if confidence > box_confidence_threshold:
                # 캠에 상자 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, org, font, fontScale, color, thickness)

        # 이벤트 발생 시 캡처 실행
        if calling_count == calling_limit:
            capture_event_image(frame, "calling")
        if smoking_count == smoking_limit:
            capture_event_image(frame, "smoking")
        if texting_count == texting_limit:
            capture_event_image(frame, "texting")

    return frame


def send_notification(class_name):
    # 여기에 실제 알림 로직을 구현하세요.
    print(f"{class_name} 감지에 대한 알림을 보냅니다.")