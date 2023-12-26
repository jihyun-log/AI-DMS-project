# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

# import tensorflow as tf
# print("TensorFlow 버전:", tf.__version__)
# print("사용 가능한 GPU 목록:", tf.config.list_physical_devices('GPU'))

# import tensorflow as tf
#
# # GPU가 사용되고 있는지 확인
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
#
# # TensorFlow가 GPU를 사용하는지 확인
# print("TensorFlow GPU 사용 여부: ", bool(physical_devices))
#
# # 현재 선택된 GPU 장치 확인
# print("현재 선택된 GPU 장치: ", physical_devices[0].name if physical_devices else "None")


# import tensorflow as tf
# from tensorflow import device_lib
#
# device_lib.list_local_devices()
# tf.config.list_physical_devices('GPU')

# import torch
# print(torch.__version__)
# print(torch.backends.mps.is_built())

# import torch
# print(torch.backends.mps.is_available())

# import torch
#
# mps_device = torch.device("mps")
#
# # MPS 장치에 바로 tensor를 생성합니다.
# x = torch.ones(5, device=mps_device)
# # 또는
# x = torch.ones(5, device="mps")
#
# # GPU 상에서 연산을 진행합니다.
# y = x * 2
#
# # 또는, 다른 장치와 마찬가지로 MPS로 이동할 수도 있습니다.
# class YourFavoriteNet:
# 	pass
#
#
# model = YourFavoriteNet()  # 어떤 모델의 객체를 생성한 뒤,
# model.to(mps_device)       # MPS 장치로 이동합니다.
#
# # 이제 모델과 텐서를 호출하면 GPU에서 연산이 이뤄집니다.
# pred = model(x)

# import torch
# mps_device = torch.device("mps")



import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 이미지 생성
img = np.full((500, 500, 3), 255, dtype=np.uint8)  # SyntaxError 및 오타 수정

text = "안녕하세요"

# 폰트 설정
font = ImageFont.truetype('/Users/bagsangbeom/PycharmProjects/DMS/maruburi/TTF/MaruBuri-Bold.ttf', 20)

# NumPy 배열을 PIL 이미지로 변환
img_pil = Image.fromarray(img)

# 이미지에 텍스트 그리기
draw = ImageDraw.Draw(img_pil)
draw.text((50, 300), text, (0, 0, 255), font=font)

# PIL 이미지를 NumPy 배열로 변환
img = np.array(img_pil)

# 이미지를 화면에 표시
cv2.imshow("img", img)
cv2.waitKey(0)  # 올바른 함수명으로 수정
cv2.destroyAllWindows()
