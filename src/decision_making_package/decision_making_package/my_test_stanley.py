#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class LaneFollow(Node):
    def __init__(self, node_name='lane_follow', start_timer=True):
        super().__init__(node_name)

        self.cv_bridge = CvBridge()

        # ===== 이미지 / 카메라 =====
        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('process_hz', 30.0)

        # ===== 지각 (차선 마스크) =====
        self.declare_parameter('white_lower', [0, 0, 190])
        self.declare_parameter('white_upper', [180, 20, 255])
        self.declare_parameter('tophat_enable', True)
        self.declare_parameter('tophat_kernel_size', 31)
        self.declare_parameter('tophat_thresh', 30)
        self.declare_parameter('min_lane_pixels', 30)

        # ===== 제어 (Stanley / 속도) =====
        self.declare_parameter('steer_k', 0.002)
        self.declare_parameter('yaw_k', 1.0)
        self.declare_parameter('max_steer', 0.6)
        self.declare_parameter('steer_smoothing_alpha', 0.35)
        self.declare_parameter('steer_slowdown_ratio', 0.35)
        self.declare_parameter('min_smooth_speed', 0.45)

        # ===== 차선 기하 (공통) =====
        self.declare_parameter('lane_width_px', 250.0)
        self.declare_parameter('min_lane_overlap_px', 50.0)
        self.declare_parameter('narrow_both_gap_px', 220.0)
        self.declare_parameter('single_lane_track_alpha', 0.80)
        self.declare_parameter('lane_width_update_alpha', 0.10)

        # ===== NORMAL 전용 =====
        self.declare_parameter('normal_single_lane_pos_angle_deg', 10.0)  # |각도|<이 값이면 최하단 x위치로 좌/우 판정

        # ===== CAR_FOLLOW(로터리) 전용 =====
        self.declare_parameter('car_follow_hold_sec', 3.0)               # phase 이탈 후 유지 시간
        self.declare_parameter('car_follow_max_speed', 0.7)              # 속도 상한
        self.declare_parameter('car_follow_gap_thresh_px', 230.0)        # 2차선 간격: 미만이면 좁음
        self.declare_parameter('car_follow_lane_width_px', 150.0)        # 단일차선 시 가상 반대차선 간격
        self.declare_parameter('car_follow_single_lane_pos_angle_deg', 7.0)  # |각도|<이 값이면 최하단 x위치로 좌/우 판정
        self.declare_parameter('car_follow_drop_angle_deg', 10.0)        # 이 각도 이상 우측기울이면 차선 삭제

        # ===== LAST_CURVE 전용 =====
        self.declare_parameter('last_curve_max_speed', 0.7)              # 속도 상한
        self.declare_parameter('last_curve_gap_thresh_px', 230.0)        # 2차선 간격: 미만이면 좁음
        self.declare_parameter('last_curve_lane_width_px', 280.0)        # 단일차선 시 가상 반대차선 간격
        self.declare_parameter('last_curve_single_lane_pos_angle_deg', 7.0)  # |각도|<이 값이면 최하단 x위치로 좌/우 판정

        # ---- 파라미터 읽기 ----
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.process_hz = float(self.get_parameter('process_hz').value)

        self.white_lower = np.array(self.get_parameter('white_lower').value, dtype=np.uint8)
        self.white_upper = np.array(self.get_parameter('white_upper').value, dtype=np.uint8)
        self.tophat_enable = bool(self.get_parameter('tophat_enable').value)
        self.tophat_kernel_size = int(self.get_parameter('tophat_kernel_size').value)
        self.tophat_thresh = int(self.get_parameter('tophat_thresh').value)
        self.min_lane_pixels = int(self.get_parameter('min_lane_pixels').value)

        self.steer_k = float(self.get_parameter('steer_k').value)
        self.yaw_k = float(self.get_parameter('yaw_k').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.steer_smoothing_alpha = float(self.get_parameter('steer_smoothing_alpha').value)
        self.steer_slowdown_ratio = float(self.get_parameter('steer_slowdown_ratio').value)
        self.min_smooth_speed = float(self.get_parameter('min_smooth_speed').value)

        self.lane_width_px = float(self.get_parameter('lane_width_px').value)
        self.min_lane_overlap_px = float(self.get_parameter('min_lane_overlap_px').value)
        self.narrow_both_gap_px = float(self.get_parameter('narrow_both_gap_px').value)
        self.single_lane_track_alpha = float(self.get_parameter('single_lane_track_alpha').value)
        self.lane_width_update_alpha = float(self.get_parameter('lane_width_update_alpha').value)

        self.normal_single_lane_pos_angle_deg = float(self.get_parameter('normal_single_lane_pos_angle_deg').value)

        self.car_follow_hold_sec = float(self.get_parameter('car_follow_hold_sec').value)
        self.car_follow_max_speed = float(self.get_parameter('car_follow_max_speed').value)
        self.car_follow_gap_thresh_px = float(self.get_parameter('car_follow_gap_thresh_px').value)
        self.car_follow_lane_width_px = float(self.get_parameter('car_follow_lane_width_px').value)
        self.car_follow_single_lane_pos_angle_deg = float(self.get_parameter('car_follow_single_lane_pos_angle_deg').value)
        self.car_follow_drop_angle_deg = float(self.get_parameter('car_follow_drop_angle_deg').value)

        self.last_curve_max_speed = float(self.get_parameter('last_curve_max_speed').value)
        self.last_curve_gap_thresh_px = float(self.get_parameter('last_curve_gap_thresh_px').value)
        self.last_curve_lane_width_px = float(self.get_parameter('last_curve_lane_width_px').value)
        self.last_curve_single_lane_pos_angle_deg = float(self.get_parameter('last_curve_single_lane_pos_angle_deg').value)

        self.tophat_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (self.tophat_kernel_size, self.tophat_kernel_size)
        )

        # ---- BEV 변환 행렬 ----
        self.src_points = np.float32([[133.5, 224.0], [506.5, 224.0], [15.5, 315.0], [624.5, 315.0]])
        self.dst_points = np.float32([[160.0, 0.0], [480.0, 0.0], [160.0, 479.0], [480.0, 479.0]])
        self.warp_mat = cv.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_warp_mat = cv.getPerspectiveTransform(self.dst_points, self.src_points)

        # ---- phase 상태 ----
        # sticky phase: 진입하면 planner가 전송을 멈춰도 hold_sec 동안 유지 (CAR_FOLLOW 전용)
        self.latest_behavior_phase = 'NORMAL'
        self.behavior_phase = 'NORMAL'
        self.sticky_phase = None
        self.sticky_last_seen_sec = None
        self.hold_sec_by_phase = {
            'CAR_FOLLOW': self.car_follow_hold_sec,
        }

        # ---- 런타임 상태 ----
        self.bgr = None
        self.warp_img0 = None
        self.filtered_img = None
        self.gaussian_sigma = 1
        self.yaw = 0.0
        self.error = 0.0
        self.steer = 0.0
        self.angular_velocity = 0.0
        self.prev_steer = None
        self.cmd_speed = 0.0
        self.prev_lfit = None
        self.prev_rfit = None
        self.last_lane_status = 'none'
        self.narrow_both_active = False
        self.narrow_both_gap = 0.0
        self.special_lane_debug = 'off'

        # ---- I/O ----
        self.phase_sub = self.create_subscription(
            String, '/behavior/phase', self.phase_cb, 10
        )
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_cb, qos_profile_sensor_data,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/stanley/cmd_vel', 10)
        self.roi_img_pub = self.create_publisher(Image, '/roi_img', 10)
        self.binary_img_pub = self.create_publisher(Image, '/binary_img', 10)
        self.debug_publisher1 = self.create_publisher(Image, '/debugging_image1', 10)
        self.debug_publisher2 = self.create_publisher(Image, '/debugging_image2', 10)

        self.timer = None
        if start_timer:
            self.start_process_timer()
        self.get_logger().info(f'ROS2 {self.get_name()} node initialized')

    def start_process_timer(self):
        if self.timer is not None:
            return
        period = 1.0 / self.process_hz if self.process_hz > 0.0 else 1.0 / 30.0
        self.timer = self.create_timer(period, self.process)

    # ============================================================
    #  콜백
    # ============================================================
    def image_cb(self, image_msg):
        try:
            bgr = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Failed to decode camera image: {exc}')
            return
        if bgr.shape[1] != self.img_width or bgr.shape[0] != self.img_height:
            bgr = cv.resize(bgr, (self.img_width, self.img_height))
        self.bgr = bgr

    def phase_cb(self, msg):
        self.latest_behavior_phase = msg.data
        if msg.data in self.hold_sec_by_phase:
            self.sticky_phase = msg.data
            self.sticky_last_seen_sec = self.get_clock().now().nanoseconds * 1e-9
            self.behavior_phase = msg.data
            return
        self._update_sticky_hold()

    def _update_sticky_hold(self):
        # sticky phase 이탈 후에도 hold_sec 동안 유지 (다른 phase 무관)
        if self.sticky_phase is None:
            self.behavior_phase = self.latest_behavior_phase
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.sticky_last_seen_sec is None:
            self.sticky_last_seen_sec = now_sec

        hold_sec = self.hold_sec_by_phase[self.sticky_phase]
        if now_sec - self.sticky_last_seen_sec >= hold_sec:
            expired = self.sticky_phase
            self.sticky_phase = None
            self.behavior_phase = (
                'NORMAL' if self.latest_behavior_phase == expired
                else self.latest_behavior_phase
            )
        else:
            self.behavior_phase = self.sticky_phase

    # ============================================================
    #  전처리
    # ============================================================
    def warpping(self, img):
        h, w = img.shape[:2]
        return cv.warpPerspective(img, self.warp_mat, (w, h))

    def gaussian_filter(self, img):
        return cv.GaussianBlur(img, (0, 0), self.gaussian_sigma)

    def white_color_filter_hsv(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.white_lower, self.white_upper)
        return cv.bitwise_and(img, img, mask=mask)

    def binary_filter(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        return binary

    def tophat_filter(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
        tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, self.tophat_kernel)
        _, mask = cv.threshold(tophat, self.tophat_thresh, 255, cv.THRESH_BINARY)
        return mask

    def roi_set(self, img):
        h = img.shape[0]
        return img[int(h * 0.85):h, :]

    # ============================================================
    #  차선 fit 유틸
    # ============================================================
    @staticmethod
    def fit_x(fit, y_values):
        y_values = np.asarray(y_values, dtype=float)
        return fit[0] * y_values + fit[1]

    def fit_distance(self, fit_a, fit_b, img_h):
        y_values = (img_h - 1) * np.array([0.15, 0.35, 0.55, 0.75, 0.95])
        distance = np.abs(self.fit_x(fit_a, y_values) - self.fit_x(fit_b, y_values))
        return float(np.median(distance))

    @staticmethod
    def fit_angle_deg(fit):
        return math.degrees(math.atan(float(fit[0])))

    def classify_single_lane(self, fit, img_h, img_w):
        # NORMAL: |각도|<threshold면 최하단 x위치, 아니면 기울기 부호로 좌/우 판정
        slope = float(fit[0])
        angle = self.fit_angle_deg(fit)
        if abs(angle) < self.normal_single_lane_pos_angle_deg:
            x_bottom = float(self.fit_x(fit, [img_h - 1])[0])
            return 'left' if x_bottom < img_w / 2.0 else 'right'
        if slope > 0.0:
            return 'right'
        if slope < 0.0:
            return 'left'
        return None

    def update_single_lane_track(self, observed_fit, side, lane_width_px=None):
        # lane_width_px=None -> 공통 lane_width_px 사용
        width = self.lane_width_px if lane_width_px is None else float(lane_width_px)
        alpha = float(np.clip(self.single_lane_track_alpha, 0.0, 1.0))
        observed_fit = np.asarray(observed_fit, dtype=float)

        if side == 'left':
            if self.prev_lfit is None:
                tracked = observed_fit.copy()
            else:
                tracked = alpha * observed_fit + (1.0 - alpha) * self.prev_lfit
            lfit = tracked
            rfit = np.array([tracked[0], tracked[1] + width])
        else:
            if self.prev_rfit is None:
                tracked = observed_fit.copy()
            else:
                tracked = alpha * observed_fit + (1.0 - alpha) * self.prev_rfit
            rfit = tracked
            lfit = np.array([tracked[0], tracked[1] - width])

        self.prev_lfit = lfit.copy()
        self.prev_rfit = rfit.copy()
        return lfit, rfit

    def update_both_lane_track(self, lfit, rfit, img_h, img_w):
        y_values = (img_h - 1) * np.array([0.25, 0.50, 0.75, 0.95])
        measured_width = float(np.median(self.fit_x(rfit, y_values) - self.fit_x(lfit, y_values)))
        if self.min_lane_overlap_px <= measured_width <= img_w * 0.85:
            alpha = float(np.clip(self.lane_width_update_alpha, 0.0, 1.0))
            self.lane_width_px = (1.0 - alpha) * self.lane_width_px + alpha * measured_width
        self.prev_lfit = np.asarray(lfit, dtype=float).copy()
        self.prev_rfit = np.asarray(rfit, dtype=float).copy()

    def _default_lane(self, img_w, width):
        lane_center = img_w / 2.0
        half = width / 2.0
        return (np.array([0.0, lane_center - half]),
                np.array([0.0, lane_center + half]))

    # ============================================================
    #  제어
    # ============================================================
    def _phase_max_speed(self):
        # 현재 behavior_phase의 속도 상한. 상한 없는 phase면 None
        if self.behavior_phase == 'CAR_FOLLOW':
            return self.car_follow_max_speed
        if self.behavior_phase == 'LAST_CURVE':
            return self.last_curve_max_speed
        return None

    def cal_steering(self, yaw, error):
        base_speed = 1.0
        wheelbase = 0.23

        steering_angle = (
            self.yaw_k * yaw
            + np.arctan2(self.steer_k * error, max(abs(base_speed), 0.01))
        )
        raw_steering_angle = float(np.clip(steering_angle, -self.max_steer, self.max_steer))

        if self.prev_steer is None:
            steering_delta = 0.0
            steering_angle = raw_steering_angle
        else:
            steering_delta = raw_steering_angle - self.prev_steer
            alpha = float(np.clip(self.steer_smoothing_alpha, 0.0, 1.0))
            steering_angle = float(np.clip(
                self.prev_steer + alpha * steering_delta, -self.max_steer, self.max_steer
            ))

        steer_change_ratio = min(abs(steering_delta) / max(abs(self.max_steer), 0.01), 1.0)
        speed_scale = 1.0 - self.steer_slowdown_ratio * steer_change_ratio
        base_speed = max(base_speed * speed_scale, self.min_smooth_speed)

        max_speed = self._phase_max_speed()
        if max_speed is not None:
            base_speed = min(base_speed, max_speed)

        self.steer = steering_angle
        self.prev_steer = steering_angle
        self.cmd_speed = float(base_speed)
        self.angular_velocity = float(base_speed * np.tan(steering_angle) / wheelbase)

        msg = Twist()
        msg.linear.x = self.cmd_speed
        msg.angular.z = self.angular_velocity
        self.cmd_vel_pub.publish(msg)

    def cal_center_line(self, lfit, rfit):
        cfit = (lfit + rfit) / 2.0
        if self.filtered_img is not None:
            h, w = self.filtered_img.shape[:2]
        else:
            h, w = 160, self.img_width
        a, b = cfit
        x_center = a * (h * 0.9) + b
        yaw = np.arctan(a)
        error = -x_center + w / 2.0
        return yaw, error

    # ============================================================
    #  CAR_FOLLOW(로터리) 차선 선택
    #    2개: 좁으면 화면상 왼쪽 / 넓으면 오른쪽 -> 1개로 축약
    #    1개: |각도|<pos -> 최하단 x위치, 우측 급기울 -> 삭제, 그 외 -> 오른쪽
    # ============================================================
    def _car_follow_lane_select(self, candidates, img_h, img):
        img_w = img.shape[1]

        if len(candidates) == 2:
            fit_a = np.asarray(candidates[0][0], dtype=float)
            fit_b = np.asarray(candidates[1][0], dtype=float)
            gap = self.fit_distance(fit_a, fit_b, img_h)
            compare_y = (img_h - 1) * 0.80
            xa = self.fit_x(fit_a, [compare_y])[0]
            xb = self.fit_x(fit_b, [compare_y])[0]

            if gap < self.car_follow_gap_thresh_px:
                keep = candidates[0] if xa <= xb else candidates[1]
                self.special_lane_debug = f'2->1 narrow gap={gap:.0f} keep left-pos'
            else:
                keep = candidates[0] if xa >= xb else candidates[1]
                self.special_lane_debug = f'2->1 wide gap={gap:.0f} keep right'
            candidates = [keep]

        if len(candidates) == 1:
            candidate = np.asarray(candidates[0][0], dtype=float)
            angle_deg = self.fit_angle_deg(candidate)

            if abs(angle_deg) < self.car_follow_single_lane_pos_angle_deg:
                x_bottom = float(self.fit_x(candidate, [img_h - 1])[0])
                side = 'left' if x_bottom < img_w / 2.0 else 'right'
                lfit, rfit = self.update_single_lane_track(candidate, side, self.car_follow_lane_width_px)
                self.last_lane_status = f'car_follow_{side}'
                self.special_lane_debug += f' | POS {side} x={x_bottom:.0f}'
                return lfit, rfit
            if angle_deg <= -self.car_follow_drop_angle_deg:
                self.special_lane_debug += f' | DROP right {angle_deg:.1f}deg'
            else:
                lfit, rfit = self.update_single_lane_track(candidate, 'right', self.car_follow_lane_width_px)
                self.last_lane_status = 'car_follow_right'
                self.special_lane_debug += f' | SET right {angle_deg:.1f}deg'
                return lfit, rfit

        # 차선 없음 -> 이전 값 유지, 없으면 직진
        if self.prev_lfit is not None and self.prev_rfit is not None:
            self.last_lane_status = 'car_follow_hold'
            if self.special_lane_debug == 'off':
                self.special_lane_debug = 'no lane -> hold'
            return self.prev_lfit.copy(), self.prev_rfit.copy()

        self.last_lane_status = 'car_follow_straight'
        if self.special_lane_debug == 'off':
            self.special_lane_debug = 'no lane -> straight'
        return self._default_lane(img_w, self.car_follow_lane_width_px)

    # ============================================================
    #  LAST_CURVE 차선 선택
    #    2개: 좁으면 화면상 왼쪽 / 넓으면 오른쪽 -> 1개로 축약
    #    1개: |각도|<pos -> 최하단 x위치, 그 외 -> 오른쪽
    # ============================================================
    def _last_curve_lane_select(self, candidates, img_h, img):
        img_w = img.shape[1]
        width = self.last_curve_lane_width_px

        if len(candidates) == 2:
            fit_a = np.asarray(candidates[0][0], dtype=float)
            fit_b = np.asarray(candidates[1][0], dtype=float)
            gap = self.fit_distance(fit_a, fit_b, img_h)
            compare_y = (img_h - 1) * 0.80
            xa = self.fit_x(fit_a, [compare_y])[0]
            xb = self.fit_x(fit_b, [compare_y])[0]

            if gap < self.last_curve_gap_thresh_px:
                keep = candidates[0] if xa <= xb else candidates[1]
                self.special_lane_debug = f'2->1 narrow gap={gap:.0f} keep left-pos'
            else:
                keep = candidates[0] if xa >= xb else candidates[1]
                self.special_lane_debug = f'2->1 wide gap={gap:.0f} keep right'
            candidates = [keep]

        if len(candidates) == 1:
            candidate = np.asarray(candidates[0][0], dtype=float)
            angle_deg = self.fit_angle_deg(candidate)

            if abs(angle_deg) < self.last_curve_single_lane_pos_angle_deg:
                x_bottom = float(self.fit_x(candidate, [img_h - 1])[0])
                side = 'left' if x_bottom < img_w / 2.0 else 'right'
                self.special_lane_debug += f' | POS {side} x={x_bottom:.0f}'
            else:
                side = 'right'
                self.special_lane_debug += f' | SET right {angle_deg:.1f}deg'
            lfit, rfit = self.update_single_lane_track(candidate, side, width)
            self.last_lane_status = f'last_curve_{side}'
            return lfit, rfit

        # 차선 없음 -> 이전 값 유지, 없으면 직진
        if self.prev_lfit is not None and self.prev_rfit is not None:
            self.last_lane_status = 'last_curve_hold'
            if self.special_lane_debug == 'off':
                self.special_lane_debug = 'no lane -> hold'
            return self.prev_lfit.copy(), self.prev_rfit.copy()

        self.last_lane_status = 'last_curve_straight'
        if self.special_lane_debug == 'off':
            self.special_lane_debug = 'no lane -> straight'
        return self._default_lane(img_w, width)

    # ============================================================
    #  NORMAL 차선 선택
    # ============================================================
    def _normal_lane_select(self, candidates, img_h, img_w):
        if len(candidates) == 2:
            if self.fit_distance(candidates[0][0], candidates[1][0], img_h) < self.min_lane_overlap_px:
                candidates = [max(candidates, key=lambda c: c[1])]

        if len(candidates) == 2:
            fit_a = np.asarray(candidates[0][0], dtype=float)
            fit_b = np.asarray(candidates[1][0], dtype=float)
            compare_y = (img_h - 1) * 0.80
            if self.fit_x(fit_a, [compare_y])[0] <= self.fit_x(fit_b, [compare_y])[0]:
                lfit, rfit = fit_a, fit_b
            else:
                lfit, rfit = fit_b, fit_a

            lane_width = self.fit_distance(lfit, rfit, img_h)
            if lane_width >= self.narrow_both_gap_px:
                self.last_lane_status = 'both'
                self.update_both_lane_track(lfit, rfit, img_h, img_w)
                return lfit, rfit

            if lane_width >= self.min_lane_overlap_px:
                # both지만 간격이 좁음 -> 다수 후보 기울기로 좌/우 재판정
                self.narrow_both_active = True
                self.narrow_both_gap = lane_width
                slope = float(np.asarray(max(candidates, key=lambda c: c[1])[0], dtype=float)[0])
                if slope > 0.0:
                    side, candidate = 'right', lfit
                elif slope < 0.0:
                    side, candidate = 'left', rfit
                else:
                    side, candidate = None, None
            else:
                candidate = max(candidates, key=lambda c: c[1])[0]
                side = self.classify_single_lane(candidate, img_h, img_w)

            if side is None:
                return self._fallback_lane(img_w)
            lfit, rfit = self.update_single_lane_track(candidate, side)
            self.last_lane_status = (
                f'narrow_both_{side}' if self.narrow_both_active else f'tracked_{side}_only'
            )
            return lfit, rfit

        if len(candidates) == 1:
            candidate = np.asarray(candidates[0][0], dtype=float)
            side = self.classify_single_lane(candidate, img_h, img_w)
            if side is None:
                return self._fallback_lane(img_w)
            lfit, rfit = self.update_single_lane_track(candidate, side)
            self.last_lane_status = f'tracked_{side}_only'
            return lfit, rfit

        if self.prev_lfit is not None and self.prev_rfit is not None:
            self.last_lane_status = 'previous'
            return self.prev_lfit.copy(), self.prev_rfit.copy()

        self.last_lane_status = 'default'
        return self._default_lane(img_w, self.lane_width_px)

    def _fallback_lane(self, img_w):
        self.last_lane_status = 'single_ambiguous'
        if self.prev_lfit is not None and self.prev_rfit is not None:
            return self.prev_lfit.copy(), self.prev_rfit.copy()
        return self._default_lane(img_w, self.lane_width_px)

    # ============================================================
    #  슬라이딩 윈도우
    # ============================================================
    def sliding_window(self, img, n_windows=10, margin=12, minpix=5):
        self._update_sticky_hold()
        self.narrow_both_active = False
        self.special_lane_debug = 'off'

        y = img.shape[0]
        histogram = np.sum(img[y // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_current = int(np.argmax(histogram[:midpoint]))
        rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint)

        window_height = int(y / n_windows)
        nz = img.nonzero()
        left_lane_inds = []
        right_lane_inds = []
        out_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        for window in range(n_windows):
            win_yl = y - (window + 1) * window_height
            win_yh = y - window * window_height
            win_xll, win_xlh = leftx_current - margin, leftx_current + margin
            win_xrl, win_xrh = rightx_current - margin, rightx_current + margin

            cv.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
            cv.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

            good_left_inds = (
                (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)
            ).nonzero()[0]
            good_right_inds = (
                (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nz[1][good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        candidates = []
        if len(left_lane_inds) >= self.min_lane_pixels:
            lfit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 1)
            candidates.append((lfit, len(left_lane_inds), 'window_left'))
        if len(right_lane_inds) >= self.min_lane_pixels:
            rfit = np.polyfit(nz[0][right_lane_inds], nz[1][right_lane_inds], 1)
            candidates.append((rfit, len(right_lane_inds), 'window_right'))

        if self.behavior_phase == 'CAR_FOLLOW':
            lfit, rfit = self._car_follow_lane_select(candidates, y, img)
        elif self.behavior_phase == 'LAST_CURVE':
            lfit, rfit = self._last_curve_lane_select(candidates, y, img)
        else:
            lfit, rfit = self._normal_lane_select(candidates, y, img.shape[1])

        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

        y_bottom = y - 1
        lt = int(np.clip(lfit[1], 0, img.shape[1] - 1))
        lb = int(np.clip(lfit[0] * y_bottom + lfit[1], 0, img.shape[1] - 1))
        rt = int(np.clip(rfit[1], 0, img.shape[1] - 1))
        rb = int(np.clip(rfit[0] * y_bottom + rfit[1], 0, img.shape[1] - 1))
        cv.line(out_img, (lt, 0), (lb, y_bottom), (255, 255, 0), 3)
        cv.line(out_img, (rt, 0), (rb, y_bottom), (0, 255, 255), 3)
        self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(out_img, encoding='bgr8'))

        return lfit, rfit

    # ============================================================
    #  디버그 오버레이
    # ============================================================
    def draw_lane(self, image, warp_roi, warp_img0, inv_mat, left_fit, right_fit):
        base_warp = warp_img0 if warp_img0 is not None else warp_roi
        full_h = base_warp.shape[0]
        roi_h = warp_roi.shape[0]
        roi_offset_y = full_h - roi_h

        ploty = np.linspace(0, roi_h - 1, roi_h)
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]
        ploty_full = ploty + roi_offset_y

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_full]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_full])))])
        pts = np.hstack((pts_left, pts_right))

        color_warp = np.zeros_like(base_warp).astype(np.uint8)
        cv.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        cv.polylines(color_warp, np.int32(pts_left), False, (255, 255, 0), 5)
        cv.polylines(color_warp, np.int32(pts_right), False, (0, 255, 255), 5)

        newwarp = cv.warpPerspective(color_warp, inv_mat, (image.shape[1], image.shape[0]))
        result = cv.addWeighted(image, 1, newwarp, 0.3, 0)

        steer_deg = math.degrees(self.steer)
        cv.putText(result, f'yaw: {self.yaw:.3f} rad / steer: {steer_deg:.1f} deg / ang_z: {self.angular_velocity:.2f}',
                   (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(result, f'err: {self.error:.1f} px / v: {self.cmd_speed:.2f}',
                   (30, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(result, f'lane: {self.last_lane_status}',
                   (30, 145), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv.LINE_AA)

        if self.narrow_both_active:
            cv.putText(result, f'narrow_both! gap={self.narrow_both_gap:.1f}px < {self.narrow_both_gap_px:.0f}px',
                       (30, 250), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv.LINE_AA)

        phase_text = f'phase: {self.behavior_phase}'
        if self.sticky_phase is not None:
            phase_text += ' (latched)'
        cv.putText(result, phase_text, (30, 285), cv.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv.LINE_AA)

        if self.behavior_phase in ('CAR_FOLLOW', 'LAST_CURVE') or self.special_lane_debug != 'off':
            cv.putText(result, f'special_lane: {self.special_lane_debug}',
                       (30, 320), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 80), 2, cv.LINE_AA)
        return result

    # ============================================================
    #  메인 루프
    # ============================================================
    def process(self):
        if self.bgr is None:
            return

        self.warp_img0 = self.warpping(self.bgr)
        warp_roi = self.roi_set(self.warp_img0)
        g_filtered = self.gaussian_filter(warp_roi)
        self.roi_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(g_filtered, encoding='bgr8'))

        hsv_binary = self.binary_filter(self.white_color_filter_hsv(g_filtered))
        if self.tophat_enable:
            self.filtered_img = cv.bitwise_and(hsv_binary, self.tophat_filter(g_filtered))
        else:
            self.filtered_img = hsv_binary
        self.binary_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.filtered_img, encoding='mono8'))

        lfit, rfit = self.sliding_window(self.filtered_img)
        self.yaw, self.error = self.cal_center_line(lfit, rfit)
        self.cal_steering(self.yaw, self.error)

        debug2_img = self.draw_lane(self.bgr, warp_roi, self.warp_img0, self.inv_warp_mat, lfit, rfit)
        self.debug_publisher2.publish(self.cv_bridge.cv2_to_imgmsg(debug2_img, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollow()
    try:
        node.get_logger().info('mission start!!! / Lane Following is always working...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()