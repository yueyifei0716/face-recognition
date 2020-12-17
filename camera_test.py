# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from cv2 import cv2


def video_demo():
    capture = cv2.VideoCapture(1)
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("video", frame)
        c = cv2.waitKey(50)
        if c == 27:
            break
    print(ret)


video_demo()
cv2.destroyAllWindows()
