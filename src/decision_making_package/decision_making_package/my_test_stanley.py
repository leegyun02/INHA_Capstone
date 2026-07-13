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
from std_msgs.msg import Bool, String


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
        self.declare_parameter('tophat_kernel_size', 40)
        self.declare_parameter('tophat_thresh', 30)
        self.declare_parameter('min_lane_pixels', 30)
        # base 탐색: 안쪽(중앙)부터 바깥으로 스캔해 임계값 넘는 첫 피크를 차선 base로 선택
        self.declare_parameter('inner_base_search', True)      # False면 기존 argmax 방식
        self.declare_parameter('base_peak_ratio', 0.5)         # 각 절반 최댓값 대비 피크 인정 비율
        self.declare_parameter('base_min_sum', 1500.0)         # 히스토그램 열 합 절대 하한(≈6px)
        self.declare_parameter('base_peak_margin', 15)         # 선택 피크 중심 보정 반경(px)

        # ===== 제어 (Stanley / 속도) =====
        self.declare_parameter('steer_k', 0.002)
        self.declare_parameter('yaw_k', 1.0)
        self.declare_parameter('max_steer', 0.6)
        self.declare_parameter('steer_smoothing_alpha', 0.35)
        self.declare_parameter('steer_slowdown_ratio', 0.35)
        self.declare_parameter('min_smooth_speed', 0.45)

        # ===== 차선 기하 (공통) =====
        self.declare_parameter('lane_width_px', 290.0)
        self.declare_parameter('min_lane_overlap_px', 50.0)
        self.declare_parameter('narrow_both_gap_px', 220.0)
        self.declare_parameter('single_lane_track_alpha', 0.80)
        self.declare_parameter('lane_width_update_alpha', 0.10)

        # ===== NORMAL 전용 =====
        self.declare_parameter('normal_single_lane_pos_angle_deg', 10.0)  # |각도|<이 값이면 최하단 x위치로 좌/우 판정

        # ===== CAR_FOLLOW(로터리) 전용 =====
        self.declare_parameter('car_follow_hold_sec', 5.0)               # phase 이탈 후 유지 시간
        self.declare_parameter('car_follow_max_speed', 1.0)              # 속도 상한
        self.declare_parameter('car_follow_gap_thresh_px', 250.0)        # 2차선 간격: 미만이면 좁음
        self.declare_parameter('car_follow_lane_width_px', 150.0)        # 차선 미검출(직진 폴백) 시 가상 차폭
        self.declare_parameter('car_follow_lane_width_left_px',450.0)   # 왼쪽 차선만 검출 시 가상 우측차선 오프셋(차폭)
        self.declare_parameter('car_follow_lane_width_right_px', 150.0)  # 오른쪽 차선만 검출 시 가상 좌측차선 오프셋(차폭)
        self.declare_parameter('car_follow_single_lane_pos_angle_deg', 7.0)  # |각도|<이 값이면 최하단 x위치로 좌/우 판정
        self.declare_parameter('car_follow_drop_angle_deg', 10.0)        # 이 각도 이상 우측기울이면 차선 삭제

        # ===== LAST_CURVE 전용 =====
        self.declare_parameter('last_curve_max_speed', 1.0)              # 속도 상한

        # ===== LAST_LANE 진입 트리거 (가로 정지선 감지, BEV 이진화 이미지 기반) =====
        self.declare_parameter('stopline_row_ratio_thresh', 0.4)   # 한 행이 이 비율 이상 흰 픽셀이면 "가로줄"로 카운트
        self.declare_parameter('stopline_min_rows', 3)              # 가로줄로 카운트된 행이 이만큼 있어야 후보로 인정
        self.declare_parameter('stopline_confirm_frames', 1)        # 이 프레임 연속 후보여야 최종 확정(latch)
        self.declare_parameter('stopline_detect_delay_sec', 3.0)    # LAST_CURVE 진입 후 이 시간 동안 정지선 감지 억제
        self.declare_parameter('last_lane_speed', 0.3)              # 확정 후 속도 상한 (last_curve_max_speed보다 우선)

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
        self.inner_base_search = bool(self.get_parameter('inner_base_search').value)
        self.base_peak_ratio = float(self.get_parameter('base_peak_ratio').value)
        self.base_min_sum = float(self.get_parameter('base_min_sum').value)
        self.base_peak_margin = int(self.get_parameter('base_peak_margin').value)

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
        self.car_follow_lane_width_left_px = float(self.get_parameter('car_follow_lane_width_left_px').value)
        self.car_follow_lane_width_right_px = float(self.get_parameter('car_follow_lane_width_right_px').value)
        self.car_follow_single_lane_pos_angle_deg = float(self.get_parameter('car_follow_single_lane_pos_angle_deg').value)
        self.car_follow_drop_angle_deg = float(self.get_parameter('car_follow_drop_angle_deg').value)

        self.last_curve_max_speed = float(self.get_parameter('last_curve_max_speed').value)

        self.stopline_row_ratio_thresh = float(self.get_parameter('stopline_row_ratio_thresh').value)
        self.stopline_min_rows = int(self.get_parameter('stopline_min_rows').value)
        self.stopline_confirm_frames = int(self.get_parameter('stopline_confirm_frames').value)
        self.stopline_detect_delay_sec = float(self.get_parameter('stopline_detect_delay_sec').value)
        self.last_lane_speed = float(self.get_parameter('last_lane_speed').value)

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
        self.both_lane_gap_px = 0.0
        self.special_lane_debug = 'off'

        # ---- LAST_LANE(가로 정지선) 감지 상태 ----
        self.stopline_hit_streak = 0
        self.last_lane_triggered = False   # 한번 확정되면 계속 유지 (sticky)
        self.last_curve_enter_sec = None   # LAST_CURVE 최초 진입 시각 (정지선 감지 억제 타이머용)

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
        self.last_lane_pub = self.create_publisher(Bool, '/lane/last_lane_detected', 10)

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
    #  LAST_LANE 진입 트리거 (가로 정지선 감지)
    #    슬라이딩 윈도우가 찾는 좌/우 차선(세로에 가까운 얇은 두 줄)과 달리,
    #    가로 정지선은 이미지 폭 전체에 걸쳐 흰 픽셀이 깔린 행(row)이 여러 줄 나온다.
    #    노이즈 방지로 stopline_confirm_frames 연속 감지돼야 최종 확정(latch)한다.
    # ============================================================
    def _detect_stop_line(self, img):
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return False
        row_white = np.count_nonzero(img > 0, axis=1)
        row_ratio = row_white / float(img.shape[1])
        hit_rows = int(np.sum(row_ratio >= self.stopline_row_ratio_thresh))
        return hit_rows >= self.stopline_min_rows

    def _update_last_lane_trigger(self, img):
        if self.last_lane_triggered:
            self.last_lane_pub.publish(Bool(data=True))
            return

        # 가로 정지선 감지는 반드시 LAST_CURVE phase 에서만 수행한다.
        # (원형 교차로 등 다른 구간의 전폭 흰 픽셀 오검출 방지)
        if self.behavior_phase != 'LAST_CURVE':
            self.stopline_hit_streak = 0
            self.last_curve_enter_sec = None
            self.last_lane_pub.publish(Bool(data=False))
            return

        # LAST_CURVE 진입 직후 stopline_detect_delay_sec 동안은 정지선 감지를 억제한다.
        # (진입 순간에 남아있는 전폭 흰 픽셀로 인한 조기 오검출 방지)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_curve_enter_sec is None:
            self.last_curve_enter_sec = now_sec
        if now_sec - self.last_curve_enter_sec < self.stopline_detect_delay_sec:
            self.stopline_hit_streak = 0
            self.last_lane_pub.publish(Bool(data=False))
            return

        img = self.binary_filter(self.white_color_filter_hsv(img))

        if self._detect_stop_line(img):
            self.stopline_hit_streak += 1
        else:
            self.stopline_hit_streak = 0

        if self.stopline_hit_streak >= self.stopline_confirm_frames:
            self.last_lane_triggered = True
            self.get_logger().warn(
                f'🛑 가로 정지선 감지 → LAST_LANE 진입, 속도 {self.last_lane_speed}로 제한'
            )

        self.last_lane_pub.publish(Bool(data=self.last_lane_triggered))

    # ============================================================
    #  제어
    # ============================================================
    def _phase_max_speed(self):
        # LAST_LANE(가로 정지선) 확정 후에는 다른 phase보다 우선해서 속도 제한
        if self.last_lane_triggered:
            return self.last_lane_speed
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
    def _car_follow_lane_width(self, side):
        # 단일차선 검출 시 가상 반대차선까지의 오프셋(차폭)을 좌/우로 구분
        if side == 'left':
            return self.car_follow_lane_width_left_px
        return self.car_follow_lane_width_right_px

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
                lfit, rfit = self.update_single_lane_track(candidate, side, self._car_follow_lane_width(side))
                self.last_lane_status = f'car_follow_{side}'
                self.special_lane_debug += f' | POS {side} x={x_bottom:.0f}'
                return lfit, rfit
            if angle_deg <= -self.car_follow_drop_angle_deg:
                self.special_lane_debug += f' | DROP right {angle_deg:.1f}deg'
            else:
                lfit, rfit = self.update_single_lane_track(candidate, 'right', self._car_follow_lane_width('right'))
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
                self.both_lane_gap_px = lane_width
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

    def _search_base_inner(self, histogram, lo, hi, from_inner_right):
        """[lo, hi) 구간에서 안쪽(중앙)부터 바깥으로 스캔해 임계값을 넘는
        첫 피크의 x 좌표를 반환. 유효 피크가 없으면 기존 argmax로 폴백."""
        seg = np.asarray(histogram[lo:hi], dtype=float)
        if seg.size == 0 or seg.max() <= 0:
            return int(np.argmax(seg)) + lo if seg.size else lo
        thresh = max(seg.max() * self.base_peak_ratio, self.base_min_sum)
        idxs = np.where(seg >= thresh)[0]
        if idxs.size == 0:
            return int(np.argmax(seg)) + lo
        start = int(idxs[-1]) if from_inner_right else int(idxs[0])  # 안쪽부터 첫 피크
        a = max(0, start - self.base_peak_margin)
        b = min(seg.size, start + self.base_peak_margin + 1)
        return int(np.argmax(seg[a:b])) + a + lo                     # 피크 중심으로 보정

    # ============================================================
    #  슬라이딩 윈도우
    # ============================================================
    def sliding_window(self, img, n_windows=10, margin=16, minpix=5):
        self._update_sticky_hold()
        self.narrow_both_active = False
        self.special_lane_debug = 'off'

        y = img.shape[0]
        histogram = np.sum(img[y // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        if self.inner_base_search:
            # 왼쪽 절반은 안쪽(오른쪽)부터, 오른쪽 절반은 안쪽(왼쪽)부터 탐색
            leftx_current = self._search_base_inner(histogram, 0, midpoint, from_inner_right=True)
            rightx_current = self._search_base_inner(histogram, midpoint, histogram.shape[0], from_inner_right=False)
        else:
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
    def draw_lane(self, image, warp_roi, warp_img0, inv_mat, left_fit, right_fit,
                  tophat_removed=None):
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

        # tophat이 제거한 픽셀을 원래 자리(원본 영상 좌표)에 빨간색으로 표시
        if tophat_removed is not None:
            removed_full = np.zeros(base_warp.shape[:2], dtype=np.uint8)
            removed_full[roi_offset_y:roi_offset_y + roi_h, :] = tophat_removed
            removed_unwarp = cv.warpPerspective(removed_full, inv_mat,
                                                (image.shape[1], image.shape[0]))
            result[removed_unwarp > 0] = (0, 0, 255)

        steer_deg = math.degrees(self.steer)
        cv.putText(result, f'yaw: {self.yaw:.3f} rad / steer: {steer_deg:.1f} deg / ang_z: {self.angular_velocity:.2f}',
                   (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(result, f'err: {self.error:.1f} px / v: {self.cmd_speed:.2f}',
                   (30, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(result, f'lane: {self.last_lane_status}',
                   (30, 145), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv.LINE_AA)

        if self.last_lane_status == 'both':
            cv.putText(result, f'both gap: {self.both_lane_gap_px:.0f} px',
                       (30, 180), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

        if self.narrow_both_active:
            cv.putText(result, f'narrow_both! gap={self.narrow_both_gap:.1f}px < {self.narrow_both_gap_px:.0f}px',
                       (30, 250), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv.LINE_AA)

        phase_text = f'phase: {self.behavior_phase}'
        if self.sticky_phase is not None:
            phase_text += ' (latched)'
        if self.last_lane_triggered:
            phase_text += ' | LAST_LANE!'
        cv.putText(result, phase_text, (30, 285), cv.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv.LINE_AA)

        if self.behavior_phase in ('CAR_FOLLOW', 'LAST_CURVE') or self.special_lane_debug != 'off':
            cv.putText(result, f'special_lane: {self.special_lane_debug}',
                       (30, 320), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 80), 2, cv.LINE_AA)

        # 주차 트리거(흰색 가로 정지선) 발동 시 상단에 눈에 띄는 배너 표시
        if self.last_lane_triggered:
            h, w = result.shape[:2]
            overlay = result.copy()
            cv.rectangle(overlay, (0, 0), (w, 70), (0, 0, 255), -1)
            cv.addWeighted(overlay, 0.5, result, 0.5, 0, result)
            banner = 'PARKING TRIGGER: STOP LINE DETECTED'
            (tw, _), _ = cv.getTextSize(banner, cv.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv.putText(result, banner, ((w - tw) // 2, 48),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv.LINE_AA)
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
        self.roi_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.warp_img0, encoding='bgr8'))

        hsv_binary = self.binary_filter(self.white_color_filter_hsv(g_filtered))
        tophat_removed = None
        # LAST_CURVE 구간에서는 tophat 로직을 적용하지 않음
        if self.tophat_enable and self.behavior_phase != 'LAST_CURVE':
            tophat_mask = self.tophat_filter(g_filtered)
            self.filtered_img = cv.bitwise_and(hsv_binary, tophat_mask)
            # tophat으로 인해 지워진 픽셀(원래 흰색이었으나 tophat이 제거한 자리)
            tophat_removed = cv.bitwise_and(hsv_binary, cv.bitwise_not(tophat_mask))
        else:
            self.filtered_img = hsv_binary
        self.binary_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.filtered_img, encoding='mono8'))

        self._update_last_lane_trigger(self.warp_img0)
        lfit, rfit = self.sliding_window(self.filtered_img)
        self.yaw, self.error = self.cal_center_line(lfit, rfit)
        self.cal_steering(self.yaw, self.error)

        debug2_img = self.draw_lane(self.bgr, warp_roi, self.warp_img0, self.inv_warp_mat, lfit, rfit,
                                    tophat_removed=tophat_removed)
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