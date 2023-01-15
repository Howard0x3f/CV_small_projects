import cv2
import pyautogui as pb
from Hand import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as nb
import math, time
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import dlib


def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    c = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (a + b) / (2.0 * c)
    return ear


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 调用摄像头
camera.set(3, 1080)  # 设置摄像头屏幕的大小
camera.set(4, 720)
hand_detector = HandDetector()
devices = AudioUtilities.GetSpeakers()  # 初始化windows音频控制对象
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # 调用系统的音频控制接口
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # 获取电脑音量范围
frameR = 50
plocx, plocy = 0, 0
finger_count = {'Left': {}, 'Right': {}}
tip = {'Left': {4: 0, 8: 0, 12: 0, 16: 0, 20: 0}, 'Right': {4: 0, 8: 0, 12: 0, 16: 0, 20: 0}}
# 获取最大最小音量
minVol = volRange[0]
maxVol = volRange[1]
space_flag =  False

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 6 #这个值越大，就需要闭眼更长的时间触发空格

# 初始化计数器
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

while True:
    success, img = camera.read()
    if img is None:
        break
    if success:
        h, w, c = img.shape  # 获取摄像头的屏幕大小
        h1, w1 = pb.size()  # 获取电脑屏幕的大小

        x, y = pb.position()

        img = cv2.flip(img, 1)

        # 开始检测闭眼
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        rects = detector(gray, 0)
        # 遍历每一个检测到的人脸
        for rect in rects:
            # 获取坐标
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # 分别计算ear值
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # 算一个平均ear
            ear = (leftEAR + rightEAR) / 2.0

            # 绘制眼睛区域
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # 检查是否满足阈值
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                space_flag = False

            # else:
                # 如果连续几帧都是闭眼的，总数算一次
            if (COUNTER >= EYE_AR_CONSEC_FRAMES and not space_flag):
                pb.press("space")
                space_flag = True
                print('space')
                # 重置
                COUNTER = 0
            elif(space_flag):
                COUNTER = 0


            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        hand_detector.hand(img, draw=False)
        position, label = hand_detector.hand_position(img)  # 获取手部的信息
        center = hand_detector.position[label].get(0, None)

        finger_count[label], tip[label] = hand_detector.finger_up(label)  # 获取竖起的手指数量和哪只手指是否竖起

        left_finger_count, tip['Left'] = hand_detector.finger_up('Left')

        cv2.putText(img, str(left_finger_count), (100, 150), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 3)

        right_finger_count, tip['Right'] = hand_detector.finger_up('Right')

        cv2.putText(img, str(right_finger_count), (w - 200, 150), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 255), 3)

        hand_length1 = hand_detector.handtips_distance(img, 8, 12, label)

        hand_length2 = hand_detector.handtips_distance(img, 4, 8, label)

        if tip[label][4] == 1 and tip[label][8] == 1 and finger_count[label] == 2:  # 当大拇指和食指竖起时，改变两指间的距离来调整音量大小
            vol = nb.interp(hand_length2, [40, 250], [minVol, maxVol])
            print(tip[label][4])
            print(vol, hand_length2)
            volume.SetMasterVolumeLevel(vol, None)

        if finger_count[label] == 5:  # 当五指全部竖起时，掌心的横坐标向左或右超过某一值时，将执行向左或右的键盘功能

            if label == 'Right':
                if center[0] > w - 300:
                    time.sleep(0.5)
                    pb.press('right')
                    print("Right")
                else:
                    if center[0] < w - 460:
                        time.sleep(0.5)
                        pb.press('left')
                        print("Left")
                    else:
                        print("center")
            else:
                if label == 'Left':
                    if center[0] < 250:
                        pb.press('left')
                        print("Left")
                    else:
                        if center[0] > 360:
                            pb.press('right')
                            print("Right")
                        else:
                            print("center")

        Right_roll = hand_detector.position[label].get(8, None)  # 得到食指的相关信息
        if Right_roll:
            rx, ry = Right_roll[0], Right_roll[1]
            if tip[label][8] == 0 and finger_count[label] == 0:  # 只有食指竖起时，执行鼠标向上滑动的功能，反之向下
                pb.scroll(-50, x, y)
                print("down")
            else:
                if tip[label][8] == 1 and finger_count[label] == 1:
                    pb.scroll(50, x, y)
                    print("up")

        Right_finger1 = hand_detector.position[label].get(8, None)
        Right_finger2 = hand_detector.position[label].get(12, None)

        if Right_finger1 and Right_finger2:
            cx = (Right_finger1[0] + Right_finger2[0]) // 2  # 得到食指和中指的中心坐标
            cy = (Right_finger1[1] + Right_finger2[1]) // 2
            tx = nb.interp(cx, (frameR, w - frameR), (0, w1))  # 转化为屏幕坐标
            ty = nb.interp(cy, (frameR, h - frameR), (0, h1))
            if tip[label][8] == 1 and tip[label][12] == 1 and finger_count[
                label] == 2:  # 当左手或右手的食指和中指竖起时，通过两指的指尖中心的坐标来控制移动鼠标
                pb.FAILSAFE = False
                tip_x = (plocx + (tx - plocx)) * 2
                tip_y = plocy + (ty - plocy)
                time.sleep(0.01)
                pb.moveTo(tip_x, tip_y)
                plocx, plocy = tip_x, tip_y  # 记录上次的位置
                if hand_length1 < 52:  # 当两指的指尖距离小于52mm时一直执行鼠标的左键单击功能
                    pb.mouseDown(button='left')  # pb.dragTo(tip_x, tip_y, button='left')
                    print("first")
                    cv2.circle(img, (cx, cy), 15, (144, 144, 144), cv2.FILLED)
                else:
                    pb.mouseUp(button='left')

            # 当大拇指，食指以及中指竖起时，执行鼠标的左键点击:
            if tip[label][4] == 1 and tip[label][8] == 1 and tip[label][12] == 1 and finger_count[label] == 3:
                pb.leftClick()
                print("Left")

            # 当食指和中指以及无名指和小拇指竖起时，执行鼠标的右键点击
            if tip[label][8] == 1 and tip[label][12] == 1 and tip[label][16] == 1 and tip[label][20] == 1 and \
                    finger_count[label] == 4:
                pb.rightClick()
                print("Right")

    cv2.imshow("window", img)  # 视频窗口
    if cv2.waitKey(1) & 0XFF == 27:  # 按下Esc按键退出
        break

camera.release()
cv2.destroyAllWindows()
