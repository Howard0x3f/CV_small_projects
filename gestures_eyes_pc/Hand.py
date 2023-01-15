import cv2
import math
import mediapipe as md


class HandDetector():
    def __init__(self):
        self.hand_detector = md.solutions.hands.Hands()
        self.length = 0
        self.label = "Right"

    def hand(self, img, draw=True):
        img_Rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 改变视屏色彩
        self.hand_date = self.hand_detector.process(img_Rgb)
        if draw:
            if self.hand_date.multi_hand_landmarks:
                for handlms in self.hand_date.multi_hand_landmarks:
                    md.solutions.drawing_utils.draw_landmarks(img, handlms, md.solutions.hands.HAND_CONNECTIONS)

    def hand_position(self, img):
        h, w, c = img.shape
        self.position = {'Left': {}, 'Right': {}}  # 定义一个字典
        if self.hand_date.multi_hand_landmarks:
            i = 0
            for point in self.hand_date.multi_handedness:
                score = point.classification[0].score
                if score > 0.8:  # 大于百分之八十是哪只手
                    self.label = point.classification[0].label
                    hand_lms = self.hand_date.multi_hand_landmarks[i].landmark
                    for id, lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)
                        self.position[self.label][id] = (x, y)
                        # print(label)    #输出是那只手
                i = i + 1
        return self.position, self.label

    def finger_up(self, hand='Left'):  # 获取竖起的手指数量和哪只手
        self.tips = {4, 8, 12, 16, 20}
        self.tips_count = {4: 0, 8: 0, 12: 0, 16: 0, 20: 0}
        for tip in self.tips:
            self.tip1 = self.position[hand].get(tip, None)
            self.tip2 = self.position[hand].get(tip - 2, None)
            if self.tip1 and self.tip2:
                if tip == 4:
                    if self.tip1[0] > self.tip2[0]:
                        if hand == 'Left':
                            self.tips_count[tip] = 1
                        else:
                            self.tips_count[tip] = 0
                    else:
                        if hand == 'Left':
                            self.tips_count[tip] = 0
                        else:
                            self.tips_count[tip] = 1
                else:
                    if self.tip1[1] > self.tip2[1]:
                        self.tips_count[tip] = 0
                    else:
                        self.tips_count[tip] = 1
        return list(self.tips_count.values()).count(1), self.tips_count

    def handtips_distance(self, img, rp1, rp2, hand='Right'):  # 计算两根手指间的距离
        self.length = 0
        # cx, cy = 0, 0
        Right_finger1 = self.position[hand].get(rp1, None)
        Right_finger2 = self.position[hand].get(rp2, None)
        if Right_finger1 and Right_finger2:
            cv2.circle(img, (Right_finger1[0], Right_finger1[1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (Right_finger2[0], Right_finger2[1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, Right_finger1, Right_finger2, (144, 0, 255))
            x1, y1 = Right_finger1[0], Right_finger1[1]
            x2, y2 = Right_finger2[0], Right_finger2[1]
            # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.length = math.hypot((x2 - x1), (y2 - y1))
        return self.length




