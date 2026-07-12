#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_planner_node.py
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING ↔ CONE_W1 → CONE_RECOVERY → CONE_HEADING_ALIGN → DRIVING
        DRIVING → TUNNEL → DRIVING
- Cone 미션 (Local Waypoint 기반):
    오도메트리 없이 오직 라이다 기준 상대 좌표 사용.
    최초 2개 감지 시 중앙콘과 측면콘의 오프셋을 기억.
    이후 지속적으로 감지되는 중앙콘 위치에 오프셋을 더해 가상의 W1(빈공간)을 실시간 추종.
    W1까지 갔던 cmd_vel을 저장했다가, 통과 후 angular.z 부호를 반대로 재생해 차선 복귀 자세를 안정화.
"""

import json
import math
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD        = '/stanley/cmd_vel'
SUB_OBSTACLES       = '/obstacles/fused'
SUB_TRAFFIC         = '/traffic_light'
SUB_IMU             = '/imu'
SUB_SCAN            = '/scan'
PUB_CMD_VEL         = '/cmd_vel'
PUB_PHASE           = '/behavior/phase'
PUB_WAYPOINT_MARKER = '/behavior/waypoints'  # RViz 웨이포인트 (laser_link 프레임)

# --- 신호등 출발 ---
USE_TRAFFIC_START = False

# --- 사람 정지 ---
PERSON_STOP_DIST  = 0.8
PERSON_WAIT_SEC   = 2.5
PERSON_PASS_SEC   = 5.0

# --- Car 추종 ---
CAR_GATE_LAT      = 0.6
CAR_STOP_DIST     = 0.40
CAR_RESUME_DIST   = 0.55
CAR_MID_DIST      = 0.80
CAR_CRUISE_DIST   = 1.50
CAR_MID_SPEED     = 0.5
CAR_MAX_CAP       = 1.0

# --- Cone 갈림길 (Local Navigation) ---
CONE_ENABLE       = True
CONE_AIM_DIST     = 1.7     # 콘 2개가 이 안에 들어오면 미션 시작 [m]
CONE_PASS_DIST    = 0.2     # 중앙콘이 차 앞 이 거리 이내로 들어오면 통과로 간주 [m]
CONE_TIMEOUT_SEC  = 6.0     # 무한루프 방지 타이머

CONE_AIM_GAIN     = 4.0     # 조향 게인 (angular.z = GAIN * 가상_ly) [1/s]
CONE_AIM_WMAX     = 5.0     # 각속도 제한 [rad/s]
CONE_SPEED_MAX    = 1.0     # 직진(조향 0) 시 최대 속도 [m/s]
CONE_SPEED_MIN    = 0.5     # 최대 조향 시 최소 속도 [m/s]
CONE_RECOVERY_SPEED_SCALE = 1.0  # 기록 속도 재생 비율
CONE_RECOVERY_W_SCALE     = 1.0  # 기록 조향 반대 재생 비율
CONE_RECOVERY_PURE_REPLAY = True # True면 recovery 중 기록 cmd_w를 정확히 반대 부호로만 재생
CONE_RECOVERY_MAX_SEC     = 4.0  # 복구 재생 최대 시간 [s]
CONE_USE_IMU_HEADING      = True # 복구 종료를 IMU yaw 기준으로 보정
CONE_HEADING_K            = 1.8  # 초기 heading 복귀 보정 게인
CONE_HEADING_WMAX         = 3.0  # heading 보정 각속도 제한 [rad/s]
CONE_HEADING_TOL          = 0.06 # 초기 heading 복귀 완료 허용 오차 [rad]
CONE_HEADING_SPEED        = 0.45 # history 끝난 뒤 heading만 맞출 때 속도 [m/s]
CONE_HEADING_ALIGN_MAX_SEC = 2.0 # replay 이후 yaw 정렬 최대 시간 [s]
CONE_LOG_SEC              = 0.5  # 콘 디버그 로그 주기 [s]

# --- Cone 이후 터널 벽 평행 보정 (/scan 기반) ---
CONE_USE_TUNNEL_SCAN_ALIGN = True
CONE_TUNNEL_ALIGN_K        = 1.2  # 벽 기울기 보정 게인
CONE_TUNNEL_ALIGN_WMAX     = 0.8  # 벽 평행 보정 각속도 제한 [rad/s]
CONE_TUNNEL_ALIGN_X_MIN    = 0.25 # 벽 fitting에 쓸 전방 최소 x [m]
CONE_TUNNEL_ALIGN_X_MAX    = 1.60 # 벽 fitting에 쓸 전방 최대 x [m]
CONE_TUNNEL_ALIGN_Y_MIN    = 0.15 # 너무 중앙에 가까운 scan 제외 [m]
CONE_TUNNEL_ALIGN_Y_MAX    = 1.20 # 너무 먼 측면 scan 제외 [m]
CONE_TUNNEL_ALIGN_MIN_PTS  = 6    # 직선 fitting 최소 점 개수

# --- Tunnel 주행 ---
TUNNEL_ENABLE     = True
TUNNEL_GAIN       = 3.0
TUNNEL_WMAX       = 3.0
TUNNEL_SPEED      = 0.8
TUNNEL_HOLD_SEC   = 1.0

# --- Class Name ---
PERSON_CLASS      = 'Person'
CAR_CLASS         = 'Car'
CONE_CLASS        = 'Cone'
# ============================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BehaviorPlannerNode(Node):
    def __init__(self):
        super().__init__('behavior_planner_cone_node')

        self.state = 'WAITING_GREEN' if USE_TRAFFIC_START else 'DRIVING'
        self.timer_target = 0.0

        # Car
        self.car_front_dist = None
        self.car_stopped = False
        self.car_log_state = 'none'

        # 신호등
        self.green_seen = False

        # Cone Local Waypoint
        self.cone_done = False
        self.cone_offset_x = 0.0  # 중앙콘 대비 W1의 X 오프셋
        self.cone_offset_y = 0.0  # 중앙콘 대비 W1의 Y 오프셋
        self.w1_local = None      # 차량 기준 W1 현재 좌표 (fx, ly)
        self.cone_cmd_history = []  # W1까지 주행한 cmd 기록: (dt, vx, wz)
        self.cone_last_record_t = None
        self.recovery_index = 0
        self.recovery_sample_elapsed = 0.0
        self.recovery_last_t = None
        self.recovery_start_t = None
        self.heading_align_start_t = None
        self.last_cone_log_t = 0.0
        self.current_yaw = None
        self.cone_start_yaw = None
        self.latest_scan = None

        # 터널
        self.tunnel_mid_y = 0.0

        # phase 발행용
        self.last_cap = None
        self._cone_was_active = False
        self.last_curve_latched = False

        self.create_subscription(Twist, SUB_CTRL_CMD, self.cmd_callback, 10)
        self.create_subscription(String, SUB_OBSTACLES, self.obstacle_callback, qos_profile_sensor_data)
        if CONE_USE_IMU_HEADING:
            self.create_subscription(Imu, SUB_IMU, self.imu_callback, qos_profile_sensor_data)
        if CONE_USE_TUNNEL_SCAN_ALIGN:
            self.create_subscription(LaserScan, SUB_SCAN, self.scan_callback, qos_profile_sensor_data)
        if USE_TRAFFIC_START:
            self.create_subscription(String, SUB_TRAFFIC, self.traffic_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.phase_pub = self.create_publisher(String, PUB_PHASE, 10)
        self.marker_pub = self.create_publisher(MarkerArray, PUB_WAYPOINT_MARKER, 10)
        
        self.get_logger().info('Behavior Planner Started (Local Waypoint + Replay Recovery Mode)')

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def traffic_callback(self, msg: String):
        if self.green_seen: return
        if msg.data == 'Green' and self.state == 'WAITING_GREEN':
            self.green_seen = True
            self.state = 'DRIVING'
            self.get_logger().info('🟢 Green Light Start Driving')

    def imu_callback(self, msg: Imu):
        self.current_yaw = yaw_from_quaternion(msg.orientation)

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def obstacle_callback(self, msg: String):
        try:
            obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._update_car_follow(obstacles)

        # 터널 진입
        tunnel_active = any(o.get('class') == 'TunnelActive' for o in obstacles)
        cone_active = self.state in ('CONE_W1', 'CONE_RECOVERY', 'CONE_HEADING_ALIGN')
        if TUNNEL_ENABLE:
            if tunnel_active and cone_active:
                self._update_tunnel_mid(obstacles)
            elif tunnel_active and self.state != 'TUNNEL':
                self.state = 'TUNNEL'
                self.timer_target = self.now_sec() + TUNNEL_HOLD_SEC
                self.get_logger().warn('🚇 Tunnel detected → TUNNEL mode')

            if self.state == 'TUNNEL':
                self._update_tunnel_mid(obstacles)
                return

        # 🚧 Cone 미션 (Local) 🚧
        if CONE_ENABLE and not self.cone_done:
            if self.state == 'DRIVING':
                self._trigger_cone_mission(obstacles)
            elif self.state == 'CONE_W1':
                self._update_cone_w1(obstacles)

        # 사람 정지
        if self.state != 'DRIVING':
            return

        for obs in obstacles:
            if obs.get('class') != PERSON_CLASS: continue
            fx = obs.get('forward_x')
            if fx and 0.0 < fx < PERSON_STOP_DIST:
                self.state = 'PERSON_STOP'
                self.timer_target = self.now_sec() + PERSON_WAIT_SEC
                self.get_logger().warn(f'Person {fx:.2f}m → wait')
                break

    def _update_tunnel_mid(self, obstacles):
        left_ly, right_ly = None, None
        for o in obstacles:
            if o.get('class') != 'TunnelWall': continue
            if o.get('side') == 'left': left_ly = o.get('lateral_y')
            elif o.get('side') == 'right': right_ly = o.get('lateral_y')

        if left_ly is not None and right_ly is not None:
            self.tunnel_mid_y = (left_ly + right_ly) / 2.0
        elif left_ly is not None: self.tunnel_mid_y = left_ly - 0.3
        elif right_ly is not None: self.tunnel_mid_y = right_ly + 0.3

    def _trigger_cone_mission(self, obstacles):
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None and 0.0 < fx <= CONE_AIM_DIST:
                cones.append((math.hypot(fx, ly), fx, ly))

        if len(cones) < 2:
            return

        # 거리순 정렬: [0]이 가장 가까운 중앙콘, [1]이 측면콘
        cones.sort(key=lambda c: c[0])
        c_dist, cx, cy = cones[0]
        s_dist, sx, sy = cones[1]

        # 측면 콘의 정반대 방향(빈 공간)으로 가기 위한 오프셋 기억
        # W1 = 2C - S = C + (C - S)
        self.cone_offset_x = cx - sx
        self.cone_offset_y = cy - sy

        self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)
        self.state = 'CONE_W1'
        self.timer_target = self.now_sec() + CONE_TIMEOUT_SEC
        self.cone_cmd_history = []
        self.cone_last_record_t = None
        self.cone_start_yaw = self.current_yaw
        yaw_text = 'none' if self.cone_start_yaw is None else f'{self.cone_start_yaw:+.2f}rad'
        self.get_logger().warn(
            f'🚧 Cone Trigger! W1=({self.w1_local[0]:.2f},{self.w1_local[1]:+.2f}), '
            f'offset=({self.cone_offset_x:.2f},{self.cone_offset_y:+.2f}), '
            f'start_yaw={yaw_text}'
        )

    def _update_cone_w1(self, obstacles):
        """매 프레임마다 중앙 콘의 위치를 다시 찾고 W1 좌표를 최신화"""
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None:
                cones.append((math.hypot(fx, ly), fx, ly))

        if not cones:
            # 콘이 시야에서 아예 사라지면 미션 완료로 간주 (지나쳤다고 판단)
            self._start_cone_recovery('cones lost')
            return

        # 가장 가까운 콘(중앙콘)의 위치를 지속적으로 추적
        cones.sort(key=lambda c: c[0])
        c_dist, cx, cy = cones[0]

        if cx < CONE_PASS_DIST:
            self._start_cone_recovery(f'center cone passed fx={cx:.2f}')
        else:
            # 실시간으로 중앙콘 위치에 기억해둔 오프셋을 더해 W1 갱신
            self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)

    def _record_cone_cmd(self, t, cmd):
        if self.cone_last_record_t is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self.cone_last_record_t)
        self.cone_last_record_t = t

        if dt > 0.3:
            dt = 0.3
        self.cone_cmd_history.append((dt, float(cmd.linear.x), float(cmd.angular.z)))

    def _start_cone_recovery(self, reason):
        self.state = 'CONE_RECOVERY'
        self.cone_done = True
        self.w1_local = None
        self.recovery_index = 0
        self.recovery_sample_elapsed = 0.0
        self.recovery_last_t = self.now_sec()
        self.recovery_start_t = self.recovery_last_t
        self.timer_target = self.recovery_last_t + CONE_RECOVERY_MAX_SEC

        total_t = sum(dt for dt, _, _ in self.cone_cmd_history)
        self.get_logger().warn(
            f'✅ Cone W1 done ({reason}) → RECOVERY replay '
            f'samples={len(self.cone_cmd_history)}, recorded_t={total_t:.2f}s'
        )

    def _start_cone_heading_align(self, reason, yaw_err, recovery_elapsed, replay_ratio):
        yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}rad'
        if yaw_err is None:
            self._finish_cone_mission(
                f'no heading ({reason})',
                yaw_text,
                recovery_elapsed,
                replay_ratio
            )
            return
        if abs(yaw_err) <= CONE_HEADING_TOL:
            self._finish_cone_mission(
                f'heading already aligned ({reason})',
                yaw_text,
                recovery_elapsed,
                replay_ratio
            )
            return

        self.state = 'CONE_HEADING_ALIGN'
        self.recovery_last_t = None
        self.recovery_start_t = None
        self.heading_align_start_t = self.now_sec()
        self.timer_target = self.heading_align_start_t + CONE_HEADING_ALIGN_MAX_SEC
        self.get_logger().info(
            f'Cone replay done ({reason}, yaw_err={yaw_text}, '
            f'elapsed={recovery_elapsed:.2f}s, replay={replay_ratio:.2f}) → HEADING_ALIGN'
        )

    def _finish_cone_mission(self, reason, yaw_text, elapsed, replay_ratio):
        self.state = 'DRIVING'
        self.cone_cmd_history = []
        self.cone_last_record_t = None
        self.recovery_last_t = None
        self.recovery_start_t = None
        self.heading_align_start_t = None
        self.cone_start_yaw = None
        self.get_logger().info(
            f'✅ Cone recovery done ({reason}, yaw_err={yaw_text}, '
            f'elapsed={elapsed:.2f}s, replay={replay_ratio:.2f}) → lane following'
        )

    def _cone_heading_error(self):
        if not CONE_USE_IMU_HEADING:
            return None
        if self.cone_start_yaw is None or self.current_yaw is None:
            return None
        return normalize_angle(self.cone_start_yaw - self.current_yaw)

    def _cone_heading_correction(self):
        yaw_err = self._cone_heading_error()
        if yaw_err is None:
            return 0.0
        return clamp(CONE_HEADING_K * yaw_err, -CONE_HEADING_WMAX, CONE_HEADING_WMAX)

    def _tunnel_scan_align_correction(self):
        scan = self.latest_scan
        if not CONE_USE_TUNNEL_SCAN_ALIGN or scan is None:
            return 0.0, None

        left_pts = []
        right_pts = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                if CONE_TUNNEL_ALIGN_X_MIN <= x <= CONE_TUNNEL_ALIGN_X_MAX:
                    ay = abs(y)
                    if CONE_TUNNEL_ALIGN_Y_MIN <= ay <= CONE_TUNNEL_ALIGN_Y_MAX:
                        if y > 0.0:
                            left_pts.append((x, y))
                        else:
                            right_pts.append((x, y))
            angle += scan.angle_increment

        candidates = []
        for side, pts in (('left', left_pts), ('right', right_pts)):
            if len(pts) < CONE_TUNNEL_ALIGN_MIN_PTS:
                continue
            n = float(len(pts))
            sx = sum(p[0] for p in pts)
            sy = sum(p[1] for p in pts)
            sxx = sum(p[0] * p[0] for p in pts)
            sxy = sum(p[0] * p[1] for p in pts)
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-6:
                continue
            slope = (n * sxy - sx * sy) / denom
            wall_yaw = math.atan(slope)
            candidates.append((len(pts), side, wall_yaw))

        if not candidates:
            return 0.0, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _count, side, wall_yaw = candidates[0]
        corr = clamp(
            CONE_TUNNEL_ALIGN_K * wall_yaw,
            -CONE_TUNNEL_ALIGN_WMAX,
            CONE_TUNNEL_ALIGN_WMAX
        )
        return corr, (side, wall_yaw)

    def _debug_cone_log(self, t, msg):
        if t - self.last_cone_log_t < CONE_LOG_SEC:
            return
        self.last_cone_log_t = t
        self.get_logger().info(msg)

    def _update_car_follow(self, obstacles):
        nearest = None
        for obs in obstacles:
            if obs.get('class') != CAR_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is None or ly is None: continue
            if fx <= 0.0 or abs(ly) > CAR_GATE_LAT: continue
            if nearest is None or fx < nearest:
                nearest = fx
        self.car_front_dist = nearest

    def _car_speed_cap(self):
        d = self.car_front_dist
        if d is None:
            self.car_stopped = False
            return None

        if self.car_stopped:
            if d > CAR_RESUME_DIST: self.car_stopped = False
            else: return 0.0
        elif d < CAR_STOP_DIST:
            self.car_stopped = True
            return 0.0

        if d < CAR_MID_DIST:
            frac = (d - CAR_STOP_DIST) / max(1e-6, CAR_MID_DIST - CAR_STOP_DIST)
            return max(0.0, frac) * CAR_MID_SPEED
        if d < CAR_CRUISE_DIST:
            frac = (d - CAR_MID_DIST) / max(1e-6, CAR_CRUISE_DIST - CAR_MID_DIST)
            return CAR_MID_SPEED + frac * (CAR_MAX_CAP - CAR_MID_SPEED)
        return None

    def _publish_waypoint_markers(self):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.w1_local is None or self.state != 'CONE_W1':
            self.marker_pub.publish(marker_array)
            return

        # W1 마커 (초록색) - 레이저 프레임 기준
        m1 = Marker()
        m1.header.frame_id = 'laser_link'
        m1.header.stamp = current_time
        m1.ns = 'waypoints'
        m1.id = 1
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        m1.pose.position.x = float(self.w1_local[0])
        m1.pose.position.y = float(self.w1_local[1])
        m1.pose.position.z = 0.0
        m1.pose.orientation.w = 1.0
        m1.scale.x = 0.4; m1.scale.y = 0.4; m1.scale.z = 0.4
        m1.color.r = 0.0; m1.color.g = 1.0; m1.color.b = 0.0; m1.color.a = 0.8
        marker_array.markers.append(m1)

        self.marker_pub.publish(marker_array)

    def cmd_callback(self, ctrl_msg: Twist):
        t = self.now_sec()
        out = Twist()

        self._publish_phase()
        self._publish_waypoint_markers()

        if self.state == 'WAITING_GREEN':
            self.cmd_pub.publish(out)
            return

        if self.state == 'TUNNEL':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('🚇 Tunnel end (time) → lane following')
                out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return
            w = clamp(TUNNEL_GAIN * self.tunnel_mid_y, -TUNNEL_WMAX, TUNNEL_WMAX)
            out.linear.x = TUNNEL_SPEED
            out.angular.z = w
            out = self._apply_car_cap(out)
            self.cmd_pub.publish(out)
            return

        # ============================================================
        # 🚧 Cone 미션 조향 제어 (W1 추종)
        # ============================================================
        if self.state == 'CONE_W1':
            if t > self.timer_target:
                self._start_cone_recovery('W1 timeout')
                self.cmd_pub.publish(out)
                return

            if self.w1_local is not None:
                # 계산된 W1의 가상 측면 오차(ly)를 향해 조향
                w = clamp(CONE_AIM_GAIN * self.w1_local[1], -CONE_AIM_WMAX, CONE_AIM_WMAX)

                # 조향각(w)의 크기에 비례하여 속도를 부드럽게 감속 (직진=MAX, 최대조향=MIN)
                steer_ratio = abs(w) / CONE_AIM_WMAX
                current_speed = CONE_SPEED_MAX - steer_ratio * (CONE_SPEED_MAX - CONE_SPEED_MIN)

                out.linear.x = current_speed
                out.angular.z = w
            else:
                out.linear.x = CONE_SPEED_MAX
                out.angular.z = 0.0

            self._record_cone_cmd(t, out)
            self._debug_cone_log(
                t,
                f'cone W1: target=({self.w1_local[0]:.2f},{self.w1_local[1]:+.2f}) '
                f'cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}, '
                f'record_n={len(self.cone_cmd_history)}'
                if self.w1_local is not None else
                f'cone W1: target=None cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'CONE_RECOVERY':
            if self.recovery_last_t is None:
                self.recovery_last_t = t
            dt = max(0.0, t - self.recovery_last_t)
            self.recovery_last_t = t
            self.recovery_sample_elapsed += dt

            while self.recovery_index < len(self.cone_cmd_history):
                sample_dt = self.cone_cmd_history[self.recovery_index][0]
                if sample_dt <= 1e-3:
                    self.recovery_index += 1
                    continue
                if self.recovery_sample_elapsed <= sample_dt:
                    break
                self.recovery_sample_elapsed -= sample_dt
                self.recovery_index += 1

            yaw_err = self._cone_heading_error()
            replay_done = self.recovery_index >= len(self.cone_cmd_history)
            timeout = t >= self.timer_target
            recovery_elapsed = 0.0 if self.recovery_start_t is None else t - self.recovery_start_t
            replay_ratio = 1.0
            if self.cone_cmd_history:
                replay_ratio = self.recovery_index / float(len(self.cone_cmd_history))

            if timeout or replay_done:
                yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}rad'
                reason = 'timeout' if timeout and not replay_done else 'replay done'
                if replay_done and not timeout:
                    self._start_cone_heading_align(reason, yaw_err, recovery_elapsed, replay_ratio)
                    if self.state == 'CONE_HEADING_ALIGN':
                        out.linear.x = CONE_HEADING_SPEED
                        out.angular.z = self._cone_heading_correction()
                    else:
                        out = self._apply_car_cap(ctrl_msg)
                else:
                    self._finish_cone_mission(reason, yaw_text, recovery_elapsed, replay_ratio)
                    out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return

            heading_corr = self._cone_heading_correction()
            tunnel_corr, tunnel_info = self._tunnel_scan_align_correction()
            _, vx, wz = self.cone_cmd_history[self.recovery_index]
            out.linear.x = vx * CONE_RECOVERY_SPEED_SCALE
            if CONE_RECOVERY_PURE_REPLAY:
                out.angular.z = clamp(
                    -wz * CONE_RECOVERY_W_SCALE,
                    -CONE_AIM_WMAX,
                    CONE_AIM_WMAX
                )
            else:
                out.angular.z = clamp(
                    (-wz * CONE_RECOVERY_W_SCALE) + heading_corr + tunnel_corr,
                    -CONE_AIM_WMAX,
                    CONE_AIM_WMAX
                )

            yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}'
            wall_text = 'none'
            if tunnel_info is not None:
                wall_text = f'{tunnel_info[0]}:{tunnel_info[1]:+.2f}'
            self._debug_cone_log(
                t,
                f'cone RECOVERY: idx={self.recovery_index}/{len(self.cone_cmd_history)}, '
                f'yaw_err={yaw_text}, wall={wall_text}, replay={replay_ratio:.2f}, '
                f'cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'CONE_HEADING_ALIGN':
            yaw_err = self._cone_heading_error()
            elapsed = 0.0 if self.heading_align_start_t is None else t - self.heading_align_start_t
            timeout = t >= self.timer_target
            yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}rad'

            if yaw_err is None or abs(yaw_err) <= CONE_HEADING_TOL or timeout:
                reason = 'heading timeout' if timeout and yaw_err is not None and abs(yaw_err) > CONE_HEADING_TOL else 'heading aligned'
                self._finish_cone_mission(reason, yaw_text, elapsed, 1.0)
                out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return

            out.linear.x = CONE_HEADING_SPEED
            out.angular.z = self._cone_heading_correction()
            self._debug_cone_log(
                t,
                f'cone HEADING_ALIGN: yaw_err={yaw_text}, '
                f'cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
            )
            self.cmd_pub.publish(out)
            return
        # ============================================================

        if self.state == 'PERSON_STOP':
            if t >= self.timer_target:
                self.state = 'PERSON_PASS'
                self.timer_target = t + PERSON_PASS_SEC
                self.get_logger().info('Person wait done → PASSING')
                out = ctrl_msg
            else:
                out.linear.x, out.angular.z = 0.0, 0.0
            self.cmd_pub.publish(out)
            return

        if self.state == 'PERSON_PASS':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('Normal Driving')
            out = self._apply_car_cap(ctrl_msg)
            self.cmd_pub.publish(out)
            return

        # 일반 주행 (Stanley 기반)
        out = self._apply_car_cap(ctrl_msg)
        self.cmd_pub.publish(out)

    def _publish_phase(self):
        if self.state in ('CONE_W1', 'CONE_RECOVERY', 'CONE_HEADING_ALIGN'):
            phase = 'CONE'
            self._cone_was_active = True
        else:
            if self._cone_was_active:
                self.last_curve_latched = True
                self._cone_was_active = False

            if self.last_curve_latched: phase = 'LAST_CURVE'
            elif self.last_cap is not None: phase = 'CAR_FOLLOW'
            else: phase = 'NORMAL'

        msg = String()
        msg.data = phase
        self.phase_pub.publish(msg)

    def _apply_car_cap(self, cmd: Twist):
        cap = self._car_speed_cap()
        self.last_cap = cap
        if cap is None:
            if self.car_log_state != 'none':
                self.car_log_state = 'none'
        elif cap == 0.0:
            if self.car_log_state != 'stop':
                self.car_log_state = 'stop'
        else:
            if self.car_log_state != 'cap':
                self.car_log_state = 'cap'

        if cap is not None and cmd.linear.x > cap:
            if cmd.linear.x > 1e-3:
                cmd.angular.z *= (cap / cmd.linear.x)
            cmd.linear.x = cap
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except RCLError:
            pass

if __name__ == '__main__':
    main()
