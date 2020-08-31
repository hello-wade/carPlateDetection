import cv2 #영상 처리 라이브러리입니다.

import sys

#GUI 라이브러리 PyQt5, 관련 함수입니다.
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

import threading #쓰레드 라이브러리입니다.

import numpy as np
import pytesseract #OCR 파이테서렉트 라이브러리입니다.
from time import sleep #프로그램 딜레이 함수입니다.

import busio #i2c핀 참조 라이브러리입니다.

#서보모터 드라이버, 서보모터 제어 라이브러리입니다.
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

i2c = busio.I2C(3, 2) #서보모터 드라이버의 SCL, SDA핀을 참조하여 i2c 핀을 설정합니다.
pca = PCA9685(i2c) #서보모터 드라이버의 i2c핀을 지정해줍니다.
pca.frequency = 50 #서보모터 PWM 주파수 설정합니다.

#서보모터 드라이버 0번 핀에 500~2500펄스의 범위로 각도를 제어하는 서보로 지정
servo0 = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
servo0.angle = 10 # 초기 서보모터 각도입니다.

def img_to_chars():
    print("Thread Running")
    global img_result
    global return_result_chars
    while True:
        if len(img_result) != 0:
            longest_idx, longest_text = -1, 0
            plate_chars = []

            #파이 테서렉트를 이용하여 번호판 이미지의 글자를 텍스트로 변환해줍니다.
            #한국어로 인식하며, psm 7(한줄의 텍스트를 읽음), oem 0(0번 레거시 엔진을 이용하여 문자 그대로를 인식)
            chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
            
            result_chars = ''
            has_digit = False
            for c in chars:
                #숫자 또는 한글이 포함되어 있는지 확인합니다.
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                    #인식된 데이터 중 순자가 하나라도 있는지 확인합니다.
                    if c.isdigit():
                        has_digit = True
                    result_chars += c
            
            plate_chars.append(result_chars)
            
            #인식된 텍스트 중 가장 긴 텍스트를 결과로 지정합니다.
            if has_digit and len(result_chars) > longest_text:
                chars = plate_chars[longest_idx]
            
            #택스트의 데이터 수가 3개 이상인 경우를 확인합니다.
            if len(result_chars) > 3:

                #번호판의 첫번째 데이터가 숫자이고, 네번쩨 데이터가 글자인 경우의 데이터를 전역변수인 return_result_chars에 저장합니다.
                if result_chars[0].isdigit() == True and result_chars[3].isdigit() == False:
                    return_result_chars = result_chars
            else:
                return_result_chars = ''
            
            #텍스트 데이터를 인식후에 번호판 이미지 데이터를 초기화 합니다.
            img_result = np.zeros((620, 480, 3), dtype=np.uint8)
            
        sleep(3)


def get_plate_img():
    global cap
    global img_result
    ret, frame = cap.read() #카메라로 입력되는 영상 데이터 읽습니다.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #색상 데이터를 제외한 그레이스케일 데이터를 이용합니다.
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #이미지의 노이즈를 줄이기 위해 가우시안불러를 이용합니다.
    img_thresh = cv2.adaptiveThreshold( #쓰레시홀드를 이용하여 일정 이상의 변화폭의 이미지 데이터를 255로 설정합니다.
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    #쓰레시홀드된 이미지 데이터 중에서 Coutours(윤곽선)을 찾습니다.
    contours, _ = cv2.findContours( 
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) #윤곽선을 감싸는 사각형의 좌표와 크기를 변수에 저장합니다.
        # contours_dict 리스트에 인식된 countur와 x, y, w, h, center_x, center_y 데이터를 저장합니다.
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    #번호판을 찾기 위한 과정을 진행합니다.
    MIN_AREA = 80 #번포판 숫자의 최소 넓이
    MIN_WIDTH, MIN_HEIGHT = 2, 8 #번호판의 숫자의 최소 폭과 높이
    MIN_RATIO, MAX_RATIO = 0.25, 1.0 #번호판 숫자의 비율

    possible_contours = [] #위 설정값에서 걸러진 윤곽선을 저장할 리스트

    cnt = 0
    for d in contours_dict:
        #미리 저장해둔 윤곽선의 데이터의 너비와 비율을 연산 합니다.
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        #설정해둔 설정값과 비교하여 조건에 만족한다면 possible_counours 리스트에 저장합니다.
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    
    def find_chars(contour_list, possible_contours):
        #첫 번째 중심과 두 번째 중심의 거리를 제한
        MAX_DIAG_MULTIPLYER = 5  # 5, 첫번째 윤곽선의 대각선 길이의 5배 이내
        #첫 번째 중심과 두 번째 중심의 각도 위치(직각사각형의 각도) 제한
        MAX_ANGLE_DIFF = 12.0  # 12.0, 임의의각 θ가 12도 이내
        #첫 번째 윤곽선과 두 번째 윤곽선의 면적 차이 제한
        MAX_AREA_DIFF = 0.3  # 0.5
        #첫 번째 윤곽선과 두 번째 윤곽선의 폭 차이 제한
        MAX_WIDTH_DIFF = 0.8
        #첫 번째 윤곽선과 두 번째 윤곽선의 높이 차이 제한
        MAX_HEIGHT_DIFF = 0.2
        #번호판으로 인식되는 윤곽선 데이터의 개수
        MIN_N_MATCHED = 3  # 3, 최소 3개 이상

        matched_result_idx = [] #인식된 결과를 인덱스 값으로 리스트에 저장

        #두개의 윤곽선 데이터를 서로 비교합니다.
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']: #같은 윤곽선은 생략합니다.
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(
                    np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) #첫 번째 중심점과 두 번째 중심점의 대각 길이를 계산합니다.
                if dx == 0:
                    angle_diff = 90 #dx가 0으로 계산되는 경우에는 90으로 예외처리합니다.
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx)) #두 윤곽선의 각도차이를 계산합니다.
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] *
                                d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                #위의 파라미터와 계산 결과를 비교하여 인덱스를 저장
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # 가장 먼져 비교했던 d1 데이터의 인덱스도 결과로 저장
            matched_contours_idx.append(d1['idx'])

            #인식된 숫자 후보군의 데이터 수가 3개 미만이면 번호판에서 제외
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            #번호판이라고 판단된다면 인식된 윤곽선의 인덱스 값(최종 후보군)을 matched_result_idx에 저장
            matched_result_idx.append(matched_contours_idx)

            #최종 후보군에 들지 못한 윤곽선들을 모아 unmatched_contour리스트에 저장합니다.
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            # 재귀를 통해 후보군에 들지 못한 윤곽선들을 최종 후보군의 윤곽선들과 다시 비교하여 최종 후보군에 추가합니다
            recursive_contour_list = find_chars(
                unmatched_contour, possible_contours)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
            break

        return matched_result_idx

    result_idx = find_chars(possible_contours, possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # 번호판 숫자로 인식된 결과(윤곽선을 감싸는 사각형)를 비디오 화면에 출력
    for r in matched_result:
        for d in r:
            cv2.rectangle(frame, pt1=(d['x'], d['y']), pt2=(
                d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)


    #숫자들이 기울어져 있는 경우에 대비하여 Affine Transform으로 정렬해주는 과정
    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) #x좌표의 위치 기준으로 순차적으로 정렬

        #번호판으로 예상되는 부분의 중심 좌표 X, Y를 계산
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        #번호판으로 예상되는 부분의 너비를 계산
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]
                       ['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        #번호판으로 예상되는 부분의 높이를 계산
        plate_height = int(sum_height / len(sorted_chars)
                           * PLATE_HEIGHT_PADDING)

        #번호판의 기울기를 계산하기 위해 직각삼각형의 높이, 빗변의 길이를 계산
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        #삼각형의 높이와 빗변의 길이를 이용하여 번호판이 기울어진 각도를 계산
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        #계산한 각도만큼 이미지를 회전
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        img_rotated = cv2.warpAffine(
            img_thresh, M=rotation_matrix, dsize=(640, 480))

        #번호판 부분의 이미지만 크롭
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

        for i, plate_img in enumerate(plate_imgs):

            #인식된 번호판의 이미지를 다시 쓰레시 홀딩을 진행
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(
                plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            #쓰레시 홀딩된 이미지에서 윤곽선을 한번 더 찾아줍니다.
            contours, _ = cv2.findContours(
                plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                area = w * h
                ratio = w / h

                #이전에 처리했던 기준에 맞는 데이터인지 최종적으로 비교 후 번호판 이미지의 좌표 데이터를 가져옵니다.
                if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                    if x < plate_min_x:
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h
            
            # 번호판 이미지의 좌표를 이용하여 번호판 이미지를 변수에 저장
            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

            # 글씨를 읽기 전에 가우시안 불러를 이용하여 노이즈를 감소
            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)

            #불러 처리된 이미지를 쓰레시 홀드를 이용하여 처리하고, 이미지의 조금의 여백을 추가
            _, img_result = cv2.threshold(
                img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_result = cv2.copyMakeBorder(
                img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return ret, frame


running = False


def run():
    global running
    global img_result
    img_result = np.zeros((620, 480, 3), dtype=np.uint8)
    global return_result_chars
    return_result_chars = ''

    global cap

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    label.resize(width*2, height)

    global plate_data # 번호판 데이터 전역변수 선언
    #번호판 데이터 사전형으로 등록 사전형 데이터의 인덱스가 키이고, 내부에 리스트 형태로 번호와, 허가 상태를 저장
    plate_data = {0 : ["584자8416", 1], 1 : ["317하8684", 0], 2: ["983사4168", 0], 
                3 : ["256바3425", 0], 4 : ["363호8953", 0], 5 : ["105가6233", 0]}
    
    #초기 번호판 상태에 따라서 버튼 텍스트의 색상을 1은 초록색 0은 빨강색으로 지정합니다.
    for plate in plate_data:
        plate_buttons[plate].setText(plate_data[plate][0])
        if plate_data[plate][1] is 0:
            plate_buttons[plate].setStyleSheet("color : #D73524")
        else:
            plate_buttons[plate].setStyleSheet("color : #297845")
    
    #OCR을 통해 이미지 데이터의 텍스트를 인식하는 쓰레드를 실행시킵니다.
    gettext_Thread = threading.Thread(target=img_to_chars)
    gettext_Thread.start()

    while running:
        #카메라에서 입력되는 이미지 데이터에서 번호판 부분을 인식하여 리턴되는 이미지 데이터를 저장합니다.
        return_ret, return_frame = get_plate_img()
        if return_ret:
            #cv의 프레임 데이터를 qt에 출력하기 위해 변환합니다.
            label.setPixmap(convert_cv_qt(return_frame))
            control_servo(return_result_chars)

        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
        # print("main_running")
    cap.release()
    print("Thread end.")

def control_servo(detected_chars):
    #쓰레드로 실행되고 있는 img_to_chars함수에서 번호판 이미지를 텍스트로 변환하는 과정이 처리되어
    #전역 변수인 return_result_chars에 저장된 데이터에 따라 결과 라벨과 서보모터를 제어하는 부분입니다.
    if detected_chars:
        for plate in plate_data:
            if plate_data[plate][0] == detected_chars:
                if plate_data[plate][1] == 1:
                    label2.setText(detected_chars)
                    label3.setText("번호판이 확인되었습니다.")
                    servo0.angle = 150
                    break
                else:
                    label2.setText(detected_chars)
                    label3.setText("허가되지 않은 번호판입니다.")
                    break
            else :
                label2.setText(detected_chars)
                label3.setText("등록되지 않은 번호판입니다.")
    else:
        label2.setText("--------")
        label2.setFont(QtGui.QFont("Sans", 30))
        label3.setText("번호판이 인식되지 않았습니다.")
        label3.setFont(QtGui.QFont("Sans", 30))
        servo0.angle = 10


def convert_cv_qt(cv_img):
    #이미지 데이터를 PyQt GUI에 출력하기 위해 변환하는 과정을 진행합니다.
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qImg)


#번호판의 상테와 버튼 색상을 바꿔주는 함수입니다.
def reverse_status(x):
    global plate_data
    if plate_data[x][1] is 0:
        plate_data[x][1] = 1
        plate_buttons[x].setStyleSheet("color : #297845")
    elif plate_data[x][1] is 1:
        plate_data[x][1] = 0
        plate_buttons[x].setStyleSheet("color : #D73524")

#프로그램을 종료할 때 처리되는 함수입니다.
def close_program():
    global running
    running = False
    win.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    gird_layout = QtWidgets.QGridLayout()
    
    win.setGeometry(0, 65, 1280, 655)
    
    label = QtWidgets.QLabel("카메라 화면")
    label.setStyleSheet("border: 3px solid black;")

    label2 = QtWidgets.QLabel("번호판 인식 화면")
    label2.setStyleSheet("border: 3px solid black;")
    label2.setAlignment(Qt.AlignCenter)

    label3 = QtWidgets.QLabel("인식 결과")
    label3.setStyleSheet("border: 3px solid black;")
    label3.setFont(QtGui.QFont("Sans", 40))
    label3.setAlignment(Qt.AlignCenter)

    plate_buttons = [QtWidgets.QPushButton("버튼1"), QtWidgets.QPushButton("버튼2"),
                    QtWidgets.QPushButton("버튼3"), QtWidgets.QPushButton("버튼4"),
                    QtWidgets.QPushButton("버튼5"), QtWidgets.QPushButton("버튼6")]
    
    for plate_button in plate_buttons:
        plate_idx = plate_buttons.index(plate_button)
        plate_buttons[plate_idx].setFont(QtGui.QFont("Sans", 18))
    
    plate_buttons[0].clicked.connect(lambda : reverse_status(0))
    plate_buttons[1].clicked.connect(lambda : reverse_status(1))
    plate_buttons[2].clicked.connect(lambda : reverse_status(2))
    plate_buttons[3].clicked.connect(lambda : reverse_status(3))
    plate_buttons[4].clicked.connect(lambda : reverse_status(4))
    plate_buttons[5].clicked.connect(lambda : reverse_status(5))

    gird_layout.addWidget(label, 0, 0, 2, 1)
    gird_layout.addWidget(label2, 0, 1, 1, 1)
    gird_layout.addWidget(label3, 1, 1, 1, 1)
    gird_layout.addWidget(plate_buttons[0], 2, 0)
    gird_layout.addWidget(plate_buttons[1], 2, 1)
    gird_layout.addWidget(plate_buttons[2], 3, 0)
    gird_layout.addWidget(plate_buttons[3], 3, 1)
    gird_layout.addWidget(plate_buttons[4], 4, 0)
    gird_layout.addWidget(plate_buttons[5], 4, 1)
    win.setLayout(gird_layout) 

    running = True
    th = threading.Thread(target=run)
    th.daemon = True
    th.start()

    win.show()


    sys.exit(app.exec_())
    app.quit()
    pca.deinit()