#导入库
import numpy as np
import cv2
import imutils
import time
import matplotlib.pyplot as plt

#设置视频路径
path= 'example.mp4'

#设置视频时长，用于绘制横坐标，单位为秒
total_time = 10

#通过颜色圈定小球大概位置
orangeLower = (11, 43, 46)
orangeUpper = (25, 255, 255)

#读取视频流
vs = cv2.VideoCapture(path)
time.sleep(2.0)

#用于记录横坐标的列表
xs = []

#循环读入视频的每一帧
while True:
    ret, frame = vs.read()
    if frame is None:
        break

    #对读取的帧改变大小，高斯滤波，并且将其转换到HSV色彩空间
    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame,  (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

   #构建针对黄色的mask，然后使用腐蚀和膨胀处理，使mask更加接近小球的真实形状
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #寻找小球边界
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #初始化小球球心坐标(x,y)
    center = None

    #只有检测至少一个轮廓时才开始后续处理
    if len(cnts) > 0:
        # 在mask中寻找最大的轮廓，然后计算它的最小周长和质心
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #只有当识别的球的半径大于一个阈值的时候才进行后续处理
        if radius > 10:
            #使用opencv在原视频上绘制
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # print(center[0])
    xs.append(center[1])
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #按下q结束进程
    if key == ord("q"):
        vs.stop()
        break

cv2.destroyAllWindows()

#生成横坐标
x = np.linspace(0, total_time, num = len(xs), endpoint=True, retstep=False, dtype=None)

#绘图
plt.title('x vs t')#纵坐标改成y vs t
plt.xlabel('time/s')
plt.ylabel('center of ball')
plt.plot(x,xs)
plt.savefig('result.jpg')
plt.show()






