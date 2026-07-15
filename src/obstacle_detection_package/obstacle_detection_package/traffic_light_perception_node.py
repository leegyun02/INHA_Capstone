#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_perception_node.py
- 고정 ROI(화면 왼쪽 신호등 영역)에서 HSV로 빨강/초록 판단
- "빨강 먼저 확인 → 초록 전환" 시 /traffic_light 에 'Green' 발행
- 초록 확정되면 스스로 카메라 구독 해제 (출발 후 연산 0)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# ============================================================
#                  튜닝 파라미터
# ============================================================
SUB_IMAGE_TOPIC = '/camera/color/image_raw'
PUB_TOPIC       = '/traffic_light'

# --- ROI (화면 비율 0~1, 신호등이 있는 왼쪽 영역) ---
ROI_X_MIN = 0.02
ROI_X_MAX = 0.22
ROI_Y_MIN = 0.30
ROI_Y_MAX = 0.58

# --- HSV 색 범위 ---
# 빨강 (H가 0 근처 + 180 근처 양쪽)
RED_LOWER1 = (0, 80, 80)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (160, 80, 80)
RED_UPPER2 = (180, 255, 255)
# 초록 (S 하한을 높여 회색/흰색/옅은 파랑 등 무채색 배경을 배제, 선명한 초록만)
GREEN_LOWER = (40, 90, 90)   # 채도 90 이상: 줄무늬 옷·바닥 같은 저채도 배경 걸러냄
GREEN_UPPER = (90, 255, 255)

# --- 판단 ---
COLOR_RATIO_THRES = 0.03     # 빨강용: ROI 대비 빨강 픽셀 비율 임계
GREEN_RATIO_THRES = 0.005    # 초록용: 과다노출 초록불이 매우 약해(~0.006~0.01) 임계 더 낮춤 (배경은 채도90 방어)
RED_CONFIRM       = 3        # 빨강 연속 프레임 (빨강 상태 확정)
GREEN_CONFIRM     = 5        # 초록 연속 프레임 (지속돼야 확정)

PUBLISH_DEBUG     = True      # 디버그 이미지 발행 여부
DEBUG_TOPIC       = '/traffic_light/debug'
# ============================================================


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__('traffic_light_perception_node')
        self.bridge = CvBridge()

        self.red_seen = False        # 빨강을 먼저 봤는가
        self.red_count = 0
        self.green_count = 0
        self.done = False

        self.image_sub = self.create_subscription(
            Image, SUB_IMAGE_TOPIC, self.image_callback, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(String, PUB_TOPIC, 10)
        if PUBLISH_DEBUG:
            self.pub_debug = self.create_publisher(Image, DEBUG_TOPIC, 10)

        self.get_logger().info('신호등 인식 노드 시작 (초록 대기 중)')

    def image_callback(self, msg):
        if self.done:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]

        # ROI 자르기
        x1, x2 = int(w * ROI_X_MIN), int(w * ROI_X_MAX)
        y1, y2 = int(h * ROI_Y_MIN), int(h * ROI_Y_MAX)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        # 빨강 비율
        red_mask = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
                    + cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        red_ratio = cv2.countNonZero(red_mask) / total

        # 초록 비율
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        green_ratio = cv2.countNonZero(green_mask) / total

        # --- 상태 판단 ---
        # 빨강 여부와 무관하게, "선명한 초록이 지속적으로" 보일 때만 출발 신호 발행.
        #   green_ratio > red_ratio : 빨강이 우세한 순간엔 초록으로 오인하지 않도록 하는 안전장치
        #   GREEN_CONFIRM 연속 프레임 : 순간 오검출 배제 (진짜 초록불은 계속 켜져 있음)
        if green_ratio > GREEN_RATIO_THRES and green_ratio > red_ratio:
            self.green_count += 1
            if self.green_count >= GREEN_CONFIRM:
                self.pub.publish(String(data='Green'))
                self.get_logger().info('🟢 초록 확정 → 출발 신호 발행, 감지 종료')
                self.done = True
                self.destroy_subscription(self.image_sub)  # 스스로 종료
        else:
            self.green_count = 0

        # 디버그
        if PUBLISH_DEBUG:
            dbg = img.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(dbg, f'WAIT-GREEN R:{red_ratio:.3f} G:{green_ratio:.3f} cnt:{self.green_count}/{GREEN_CONFIRM}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            try:
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()