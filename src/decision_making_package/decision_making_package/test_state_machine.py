#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_state_machine.py
- state_machine_node.py 를 베이스로, state_machine_cone_node.py 의
  '라바콘 회피 후 복구' 로직만 추가한 버전.
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING ↔ CONE_W1 → CONE_RECOVERY → CONE_HEADING_ALIGN → CONE_EXIT → DRIVING
        DRIVING → TUNNEL → DRIVING
        DRIVING → PARKING_APPROACH → PARKING_REPLAY → PARKING_DONE (종료 상태)
- Cone 미션 (Local Waypoint 기반):
    오도메트리 없이 오직 라이다 기준 상대 좌표 사용.
    최초 2개 감지 시 중앙콘과 측면콘의 오프셋을 기억.
    이후 지속적으로 감지되는 중앙콘 위치에 오프셋을 더해 가상의 W1(빈공간)을 실시간 추종.
    W1까지 갔던 cmd_vel을 저장했다가, 통과 후 angular.z 부호를 반대로 재생해 차선 복귀 자세를 안정화.
- 평행주차 미션 (cmd_vel_record_replay_node.py 를 이 상태머신에 통합한 버전):
    my_test_stanley.py 가 가로 정지선(코스 마지막 구간)을 감지하면 /lane/last_lane_detected
    (std_msgs/Bool) 를 True로 발행한다. 이 노드는 DRIVING 중 그 신호를 받으면 PARKING_APPROACH로
    전환해 라이다 전방 섹터를 보며 approach_speed로 직진 접근하고, front_trigger_dist 이내로
    들어오면 PARKING_REPLAY로 전환해 PARKING_RECORD_FILE(cmd_vel_record.json)에 저장된
    (dt, vx, wz) 시퀀스를 그대로 재생한다. 재생이 끝나면 PARKING_DONE(정차 유지, 종료 상태)으로
    남는다. 트리거는 노드당 한 번만 발동한다(parking_triggered).
"""

import json
import math
import os
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD        = '/stanley/cmd_vel'
SUB_OBSTACLES       = '/obstacles/fused'
SUB_TRAFFIC         = '/traffic_light'
SUB_IMU             = '/imu'
SUB_SCAN            = '/scan'                     # 주차 자동 접근용 라이다 입력
SUB_LAST_LANE       = '/lane/last_lane_detected'  # my_test_stanley.py 가 발행하는 가로 정지선 감지 결과
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
CONE_AIM_DIST     = 1.55    # 콘 2개가 이 안에 들어오면 미션 시작 [m]
CONE_PASS_DIST    = 0.2     # 중앙콘이 차 앞 이 거리 이내로 들어오면 통과로 간주 [m]
CONE_W1_PASS_DIST = 0.30    # W1 자체가 차 앞 이 거리 이내로 들어오면 통과로 간주 [m]
CONE_W1_STUCK_SEC = 0.8     # W1이 이 시간 동안 가까워지지 않으면 recovery로 전환 [s]
CONE_W1_PROGRESS_EPS = 0.05 # W1 x가 이만큼 줄어야 진행 중으로 인정 [m]
CONE_TIMEOUT_SEC  = 10.0     # 무한루프 방지 타이머

CONE_AIM_GAIN     = 4.0     # 조향 게인 (angular.z = GAIN * 가상_ly) [1/s]
CONE_AIM_WMAX     = 3.0     # 각속도 제한 [rad/s]
CONE_SPEED_MAX    = 0.85    # 직진(조향 0) 시 최대 속도 [m/s]
CONE_SPEED_MIN    = 0.3     # 최대 조향 시 최소 속도 [m/s]

# --- Cone 복구(Recovery) 재생 ---
CONE_RECOVERY_SPEED_SCALE = 1.0  # 기록 속도 재생 비율
CONE_RECOVERY_W_SCALE     = 1.0  # 기록 조향 반대 재생 비율
CONE_RECOVERY_RIGHT_SCALE = 2.0 # 오른쪽 복구(cmd_w 음수) 조향 추가 배율
CONE_RECOVERY_LEFT_SCALE  = 1.0  # 왼쪽 복구(cmd_w 양수) 조향 추가 배율
CONE_RECOVERY_SIGN_LOCK   = True # True면 recovery 조향 방향을 회피 시작 방향의 반대로 고정
CONE_RECOVERY_SIGN_MIN_W  = 0.15 # recovery 방향 판단에 사용할 최소 조향 [rad/s]
CONE_RECOVERY_PURE_REPLAY = True # True면 recovery 중 기록 cmd_w를 정확히 반대 부호로만 재생
CONE_RECOVERY_MAX_SEC     = 6.0  # 복구 안전 타이머 [s] (history가 있으면 replay를 우선 끝까지 수행)
CONE_USE_IMU_HEADING      = True # 복구 종료를 IMU yaw 기준으로 보정
CONE_HEADING_K            = 1.8  # 초기 heading 복귀 보정 게인
CONE_HEADING_WMAX         = 3.0  # heading 보정 각속도 제한 [rad/s]
CONE_HEADING_TOL          = 0.06 # 초기 heading 복귀 완료 허용 오차 [rad]
CONE_HEADING_SPEED        = 0.45 # history 끝난 뒤 heading만 맞출 때 속도 [m/s]
CONE_HEADING_ALIGN_MAX_SEC = 3.0 # replay 이후 yaw 정렬 최대 시간 [s]
CONE_EXIT_HOLD_SEC        = 1.0  # yaw 정렬 후 lane follower로 넘기기 전 최소 heading 유지 시간 [s]
CONE_EXIT_MAX_SEC         = 2.5  # CONE_EXIT 최대 유지 시간 [s]
CONE_EXIT_TOL             = 0.12 # CONE_EXIT 종료 yaw 허용 오차 [rad]
CONE_EXIT_SPEED           = 0.35 # CONE_EXIT 중 전진 속도 [m/s]
CONE_EXIT_WMAX            = 0.6  # CONE_EXIT 중 yaw 유지 보정 각속도 제한 [rad/s]
CONE_LOG_SEC              = 0.5  # 콘 디버그 로그 주기 [s]

# --- Cone 기준 진행방향 보정 ---
CONE_USE_PAIR_HEADING      = True # 두 콘을 잇는 선의 수직 방향을 탈출 후 진행 yaw 기준으로 사용
CONE_PAIR_HEADING_ALPHA    = 0.35 # 두 콘 heading 갱신 low-pass 비율
CONE_PAIR_HEADING_MIN_SEP  = 0.20 # heading 계산에 사용할 두 콘 최소 거리 [m]
CONE_PAIR_HEADING_MAX_SEP  = 1.50 # heading 계산에 사용할 두 콘 최대 거리 [m]
CONE_PAIR_HEADING_MAX_ABS  = 0.90 # 너무 틀어진 cone heading은 오검출로 보고 무시 [rad]

# --- Tunnel 주행 ---
TUNNEL_ENABLE     = True
TUNNEL_GAIN       = 3.0
TUNNEL_WMAX       = 3.0
TUNNEL_SPEED      = 0.8
TUNNEL_HOLD_SEC   = 1.0
PRE_CAR_FOLLOW_SEC = 0.0    # 터널 종료 직후 PRE_CAR_FOLLOW(=CAR_FOLLOW) 유지 시간 [s], 0이면 비활성

# --- LAST_CURVE 테스트용 ---
LAST_CURVE_TEST_TIMEOUT_SEC = 10.0  # [테스트용] LAST_CURVE phase 진입 후 이 시간 지나면 강제로 NORMAL 복귀

# --- 평행주차 (cmd_vel_record_replay_node.py 통합) ---
PARKING_ENABLE              = True
PARKING_RECORD_FILE         = os.path.expanduser('~/trinity_ws/cmd_vel_record.json')
PARKING_FRONT_ANGLE_CENTER      = 0.0                 # 전방 섹터 중심 각도 (REP103: 0=전방)
PARKING_FRONT_ANGLE_HALF_WIDTH  = math.radians(5)     # 전방 섹터 폭
PARKING_FRONT_TRIGGER_DIST  = 0.35   # 전방벽이 이 거리[m] 이하가 되면 자동으로 REPLAY 시작
PARKING_APPROACH_SPEED      = 0.3    # front_trigger_dist 도달 전까지 직진 접근 속도 [m/s]

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
        super().__init__('test_state_machine_node')

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

        # Cone 복구(Recovery) 재생용
        self.cone_cmd_history = []  # W1까지 주행한 cmd 기록: (dt, vx, wz)
        self.cone_last_record_t = None
        self.cone_pending_cmd = None
        self.w1_min_x = None
        self.w1_last_progress_t = None
        self.recovery_index = 0
        self.recovery_sample_elapsed = 0.0
        self.recovery_last_t = None
        self.recovery_start_t = None
        self.recovery_recorded_t = 0.0
        self.recovery_w_sign = 0.0
        self.heading_align_start_t = None
        self.cone_exit_start_t = None
        self.last_cone_log_t = 0.0
        self.current_yaw = None
        self.cone_start_yaw = None
        self.cone_target_yaw = None
        self.cone_pair_yaw_err = None

        # 터널
        self.tunnel_mid_y = 0.0

        # 평행주차
        self.latest_scan = None
        self.last_lane_detected = False
        self.parking_triggered = False
        self.parking_records = []          # [(dt, vx, wz), ...]
        self.parking_replay_index = 0
        self.parking_replay_elapsed = 0.0
        self.parking_last_t = None

        # phase 발행용
        self.last_cap = None
        self._cone_was_active = False
        self.last_curve_latched = False
        self._tunnel_was_active = False
        self.pre_car_follow_target = None   # 터널 종료 후 PRE_CAR_FOLLOW 만료 시각 (None=비활성)
        self.last_curve_start_t = None

        self.create_subscription(Twist, SUB_CTRL_CMD, self.cmd_callback, 10)
        self.create_subscription(String, SUB_OBSTACLES, self.obstacle_callback, qos_profile_sensor_data)
        if CONE_USE_IMU_HEADING:
            self.create_subscription(Imu, SUB_IMU, self.imu_callback, qos_profile_sensor_data)
        if USE_TRAFFIC_START:
            self.create_subscription(String, SUB_TRAFFIC, self.traffic_callback, 10)
        if PARKING_ENABLE:
            self.create_subscription(LaserScan, SUB_SCAN, self.scan_callback, qos_profile_sensor_data)
            self.create_subscription(Bool, SUB_LAST_LANE, self.last_lane_callback, 10)

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

    # ============================================================
    #  🅿️ 평행주차 (cmd_vel_record_replay_node.py 통합)
    # ============================================================
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def last_lane_callback(self, msg: Bool):
        if msg.data and not self.last_lane_detected:
            self.last_lane_detected = True
            self.get_logger().info('👀 last_lane_detected 수신 → 주차 접근 트리거 대기 중')

    def _front_distance(self):
        scan = self.latest_scan
        if scan is None:
            return None
        dists = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                da = PARKING_FRONT_ANGLE_CENTER - angle
                while da > math.pi:
                    da -= 2.0 * math.pi
                while da < -math.pi:
                    da += 2.0 * math.pi
                if abs(da) <= PARKING_FRONT_ANGLE_HALF_WIDTH:
                    dists.append(r)
            angle += scan.angle_increment
        if not dists:
            return None
        dists.sort()
        return dists[len(dists) // 2]

    def _load_parking_records(self):
        try:
            with open(PARKING_RECORD_FILE, 'r') as f:
                data = json.load(f)
            records = []
            for r in data.get('records', []):
                if isinstance(r, dict):
                    records.append((float(r['dt']), float(r['vx']), float(r['wz'])))
                else:
                    dt, vx, wz = r
                    records.append((float(dt), float(vx), float(wz)))
            self.get_logger().info(f'주차 기록 파일 로드: {PARKING_RECORD_FILE} ({len(records)} samples)')
            return records
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().error(f'주차 기록 파일 로드 실패: {e}')
            return []

    def _start_parking_approach(self):
        self.parking_triggered = True
        self.parking_records = self._load_parking_records()
        if not self.parking_records:
            self.get_logger().error(
                f'주차 기록이 없어 트리거 무시 ({PARKING_RECORD_FILE} 확인 필요)'
            )
            return
        self.state = 'PARKING_APPROACH'
        self.get_logger().warn(
            f'🅿️ last_lane_detected → PARKING_APPROACH 시작 (target={PARKING_FRONT_TRIGGER_DIST}m)'
        )

    def _start_parking_replay(self, front):
        self.parking_replay_index = 0
        self.parking_replay_elapsed = 0.0
        self.parking_last_t = self.now_sec()
        self.state = 'PARKING_REPLAY'
        total_t = sum(dt for dt, _, _ in self.parking_records)
        self.get_logger().warn(
            f'▶ PARKING_REPLAY 시작 (front={front:.2f}m): samples={len(self.parking_records)}, '
            f'total={total_t:.2f}s'
        )

    def obstacle_callback(self, msg: String):
        if self.state in ('PARKING_APPROACH', 'PARKING_REPLAY', 'PARKING_DONE'):
            return
        try:
            obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._update_car_follow(obstacles)

        # 터널 진입
        tunnel_active = any(o.get('class') == 'TunnelActive' for o in obstacles)
        cone_active = self.state in ('CONE_W1', 'CONE_RECOVERY', 'CONE_HEADING_ALIGN', 'CONE_EXIT')
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
        self._update_cone_pair_heading([(cx, cy), (sx, sy)])
        self.state = 'CONE_W1'
        self.timer_target = self.now_sec() + CONE_TIMEOUT_SEC
        self.car_front_dist = None
        self.car_stopped = False
        self.last_cap = None
        self.pre_car_follow_target = None
        self.cone_cmd_history = []
        self.cone_last_record_t = None
        self.cone_pending_cmd = None
        self.w1_min_x = self.w1_local[0]
        self.w1_last_progress_t = self.now_sec()
        self.cone_start_yaw = self.current_yaw
        yaw_text = 'none' if self.cone_start_yaw is None else f'{self.cone_start_yaw:+.2f}rad'
        cone_yaw_text = 'none' if self.cone_pair_yaw_err is None else f'{self.cone_pair_yaw_err:+.2f}rad'
        self.get_logger().warn(
            f'🚧 Cone Trigger! W1=({self.w1_local[0]:.2f},{self.w1_local[1]:+.2f}), '
            f'offset=({self.cone_offset_x:.2f},{self.cone_offset_y:+.2f}), '
            f'start_yaw={yaw_text}, cone_yaw={cone_yaw_text}'
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
        self._update_cone_pair_heading([(fx, ly) for _, fx, ly in cones[:2]])

        if cx < CONE_PASS_DIST:
            self._start_cone_recovery(f'center cone passed fx={cx:.2f}')
        else:
            # 실시간으로 중앙콘 위치에 기억해둔 오프셋을 더해 W1 갱신
            self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)
            self._check_w1_done('scan update')

    def _check_w1_done(self, source):
        if self.state != 'CONE_W1' or self.w1_local is None:
            return False

        t = self.now_sec()
        wx = self.w1_local[0]
        if wx <= CONE_W1_PASS_DIST:
            self._start_cone_recovery(f'W1 passed {source} x={wx:.2f}')
            return True

        if self.w1_min_x is None or wx < self.w1_min_x - CONE_W1_PROGRESS_EPS:
            self.w1_min_x = wx
            self.w1_last_progress_t = t
            return False

        if self.w1_last_progress_t is not None and t - self.w1_last_progress_t >= CONE_W1_STUCK_SEC:
            self._start_cone_recovery(
                f'W1 no progress {source} x={wx:.2f}, min_x={self.w1_min_x:.2f}'
            )
            return True

        return False

    def _cone_pair_heading_error_from_points(self, p0, p1):
        if not CONE_USE_PAIR_HEADING:
            return None

        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        sep = math.hypot(dx, dy)
        if sep < CONE_PAIR_HEADING_MIN_SEP or sep > CONE_PAIR_HEADING_MAX_SEP:
            return None

        # 두 콘을 잇는 선에 수직인 두 방향 중 차량 전방(+x)을 향하는 쪽을 진행 방향으로 본다.
        hx = dy
        hy = -dx
        if hx < 0.0:
            hx = -hx
            hy = -hy
        if hx <= 1e-6:
            return None

        yaw_err = math.atan2(hy, hx)
        if abs(yaw_err) > CONE_PAIR_HEADING_MAX_ABS:
            return None
        return yaw_err

    def _update_cone_pair_heading(self, points):
        if len(points) < 2 or self.current_yaw is None:
            return

        yaw_err = self._cone_pair_heading_error_from_points(points[0], points[1])
        if yaw_err is None:
            return

        target_yaw = normalize_angle(self.current_yaw + yaw_err)
        if self.cone_target_yaw is None:
            self.cone_target_yaw = target_yaw
        else:
            prev_err = normalize_angle(target_yaw - self.cone_target_yaw)
            self.cone_target_yaw = normalize_angle(
                self.cone_target_yaw + CONE_PAIR_HEADING_ALPHA * prev_err
            )
        self.cone_pair_yaw_err = normalize_angle(self.cone_target_yaw - self.current_yaw)

    def _record_cone_cmd(self, t, cmd):
        if self.cone_last_record_t is None:
            self.cone_last_record_t = t
            self.cone_pending_cmd = (float(cmd.linear.x), float(cmd.angular.z))
            return

        dt = max(0.0, t - self.cone_last_record_t)
        if dt > 0.3:
            dt = 0.3

        if self.cone_pending_cmd is not None and dt > 1e-3:
            vx, wz = self.cone_pending_cmd
            self.cone_cmd_history.append((dt, vx, wz))

        self.cone_last_record_t = t
        self.cone_pending_cmd = (float(cmd.linear.x), float(cmd.angular.z))

    def _finalize_cone_cmd_history(self, t):
        if self.cone_last_record_t is None or self.cone_pending_cmd is None:
            return

        dt = max(0.0, t - self.cone_last_record_t)
        if dt > 0.3:
            dt = 0.3
        if dt > 1e-3:
            vx, wz = self.cone_pending_cmd
            self.cone_cmd_history.append((dt, vx, wz))

        self.cone_last_record_t = None
        self.cone_pending_cmd = None

    def _recorded_cone_cmd_count(self):
        if self.cone_pending_cmd is None:
            return len(self.cone_cmd_history)
        else:
            return len(self.cone_cmd_history) + 1

    def _recovery_sign_from_history(self):
        for _dt, _vx, wz in self.cone_cmd_history:
            if abs(wz) >= CONE_RECOVERY_SIGN_MIN_W:
                return -1.0 if wz > 0.0 else 1.0
        return 0.0

    def _recovery_w_from_sample(self, wz):
        if CONE_RECOVERY_SIGN_LOCK and self.recovery_w_sign != 0.0:
            scale = CONE_RECOVERY_LEFT_SCALE if self.recovery_w_sign > 0.0 else CONE_RECOVERY_RIGHT_SCALE
            return self.recovery_w_sign * abs(wz) * CONE_RECOVERY_W_SCALE * scale

        base_w = -wz * CONE_RECOVERY_W_SCALE
        scale = CONE_RECOVERY_LEFT_SCALE if base_w > 0.0 else CONE_RECOVERY_RIGHT_SCALE
        return base_w * scale

    def _start_cone_recovery(self, reason):
        self._finalize_cone_cmd_history(self.now_sec())
        self.state = 'CONE_RECOVERY'
        self.cone_done = True
        self.w1_local = None
        self.w1_min_x = None
        self.w1_last_progress_t = None
        self.recovery_index = 0
        self.recovery_sample_elapsed = 0.0
        now = self.now_sec()
        self.recovery_last_t = None
        self.recovery_start_t = now
        self.timer_target = now + CONE_RECOVERY_MAX_SEC

        total_t = sum(dt for dt, _, _ in self.cone_cmd_history)
        self.recovery_recorded_t = total_t
        self.recovery_w_sign = self._recovery_sign_from_history()
        sign_text = 'none' if self.recovery_w_sign == 0.0 else ('left' if self.recovery_w_sign > 0.0 else 'right')
        self.get_logger().warn(
            f'✅ Cone W1 done ({reason}) → RECOVERY replay '
            f'samples={len(self.cone_cmd_history)}, recorded_t={total_t:.2f}s, '
            f'recovery_dir={sign_text}'
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
        self.state = 'CONE_EXIT'
        self.cone_cmd_history = []
        self.cone_last_record_t = None
        self.cone_pending_cmd = None
        self.w1_min_x = None
        self.w1_last_progress_t = None
        self.recovery_last_t = None
        self.recovery_start_t = None
        self.recovery_recorded_t = 0.0
        self.recovery_w_sign = 0.0
        self.heading_align_start_t = None
        self.cone_exit_start_t = self.now_sec()
        self.timer_target = self.cone_exit_start_t + CONE_EXIT_MAX_SEC
        self.get_logger().info(
            f'✅ Cone recovery done ({reason}, yaw_err={yaw_text}, '
            f'elapsed={elapsed:.2f}s, replay={replay_ratio:.2f}) → cone exit hold'
        )

    def _finish_cone_exit(self, reason):
        self.state = 'DRIVING'
        self.cone_start_yaw = None
        self.cone_target_yaw = None
        self.cone_pair_yaw_err = None
        self.cone_exit_start_t = None
        self.get_logger().info(f'✅ Cone exit done ({reason}) → lane following')

    def _cone_heading_error(self):
        if not CONE_USE_IMU_HEADING:
            return None
        if self.current_yaw is None:
            return None
        if CONE_USE_PAIR_HEADING and self.cone_target_yaw is not None:
            self.cone_pair_yaw_err = normalize_angle(self.cone_target_yaw - self.current_yaw)
            return self.cone_pair_yaw_err
        if self.cone_start_yaw is None:
            return None
        return normalize_angle(self.cone_start_yaw - self.current_yaw)

    def _cone_heading_source(self):
        if CONE_USE_PAIR_HEADING and self.cone_target_yaw is not None:
            return 'cone_pair'
        return 'start_yaw'

    def _cone_heading_correction(self):
        yaw_err = self._cone_heading_error()
        if yaw_err is None:
            return 0.0
        return clamp(CONE_HEADING_K * yaw_err, -CONE_HEADING_WMAX, CONE_HEADING_WMAX)

    def _cone_exit_heading_correction(self):
        yaw_err = self._cone_heading_error()
        if yaw_err is None:
            return 0.0
        return clamp(CONE_HEADING_K * yaw_err, -CONE_EXIT_WMAX, CONE_EXIT_WMAX)

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

        if PARKING_ENABLE and self.state == 'DRIVING' and self.last_lane_detected \
                and self.last_curve_latched and not self.parking_triggered:
            self._start_parking_approach()

        self._publish_phase()
        self._publish_waypoint_markers()

        if self.state == 'WAITING_GREEN':
            self.cmd_pub.publish(out)
            return

        if self.state == 'PARKING_APPROACH':
            front = self._front_distance()
            if front is not None and front <= PARKING_FRONT_TRIGGER_DIST:
                self._start_parking_replay(front)
                self.cmd_pub.publish(Twist())
                return
            if self.latest_scan is None:
                self.get_logger().warn(
                    'PARKING_APPROACH: /scan 수신 대기 중, 정지 유지', throttle_duration_sec=2.0
                )
                self.cmd_pub.publish(Twist())
                return
            out.linear.x = PARKING_APPROACH_SPEED
            self.get_logger().info(
                f'PARKING_APPROACH: front={"n/a" if front is None else f"{front:.2f}m"}, '
                f'목표={PARKING_FRONT_TRIGGER_DIST}m',
                throttle_duration_sec=1.0,
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'PARKING_REPLAY':
            dt = 0.0 if self.parking_last_t is None else max(0.0, t - self.parking_last_t)
            self.parking_last_t = t
            self.parking_replay_elapsed += dt

            while self.parking_replay_index < len(self.parking_records):
                sample_dt = self.parking_records[self.parking_replay_index][0]
                if sample_dt <= 1e-3:
                    self.parking_replay_index += 1
                    continue
                if self.parking_replay_elapsed <= sample_dt:
                    break
                self.parking_replay_elapsed -= sample_dt
                self.parking_replay_index += 1

            if self.parking_replay_index >= len(self.parking_records):
                self.state = 'PARKING_DONE'
                self.cmd_pub.publish(Twist())
                self.get_logger().warn('🅿️ PARKING 완료 (REPLAY 종료) → 정차 유지')
                return

            _, vx, wz = self.parking_records[self.parking_replay_index]
            out.linear.x = vx
            out.angular.z = wz
            self.cmd_pub.publish(out)
            return

        if self.state == 'PARKING_DONE':
            self.cmd_pub.publish(Twist())
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
                if self._check_w1_done('cmd loop'):
                    self.cmd_pub.publish(out)
                    return

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
                f'record_n={self._recorded_cone_cmd_count()}'
                if self.w1_local is not None else
                f'cone W1: target=None cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'CONE_RECOVERY':
            if self.recovery_last_t is None:
                dt = 0.0
                self.recovery_last_t = t
            else:
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
            replay_time = min(
                self.recovery_recorded_t,
                sum(dt for dt, _, _ in self.cone_cmd_history[:self.recovery_index]) +
                self.recovery_sample_elapsed
            )

            if replay_done or (timeout and not self.cone_cmd_history):
                yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}rad'
                reason = 'timeout' if timeout and not replay_done else 'replay done'
                if replay_done:
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
            _, vx, wz = self.cone_cmd_history[self.recovery_index]
            out.linear.x = vx * CONE_RECOVERY_SPEED_SCALE
            if CONE_RECOVERY_PURE_REPLAY:
                out.angular.z = clamp(
                    self._recovery_w_from_sample(wz),
                    -CONE_AIM_WMAX,
                    CONE_AIM_WMAX
                )
            else:
                out.angular.z = clamp(
                    self._recovery_w_from_sample(wz) + heading_corr,
                    -CONE_AIM_WMAX,
                    CONE_AIM_WMAX
                )

            yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}'
            self._debug_cone_log(
                t,
                f'cone RECOVERY: idx={self.recovery_index}/{len(self.cone_cmd_history)}, '
                f'yaw_err={yaw_text}, heading_src={self._cone_heading_source()}, '
                f'replay={replay_ratio:.2f}, replay_t={replay_time:.2f}/{self.recovery_recorded_t:.2f}s, '
                f'raw_w={wz:+.2f}, cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
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
                f'heading_src={self._cone_heading_source()}, '
                f'cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'CONE_EXIT':
            yaw_err = self._cone_heading_error()
            yaw_text = 'none' if yaw_err is None else f'{yaw_err:+.2f}rad'
            elapsed = 0.0 if self.cone_exit_start_t is None else t - self.cone_exit_start_t
            min_hold_done = elapsed >= CONE_EXIT_HOLD_SEC
            yaw_ready = yaw_err is None or abs(yaw_err) <= CONE_EXIT_TOL
            timeout = t >= self.timer_target
            if (min_hold_done and yaw_ready) or timeout:
                if timeout and not yaw_ready:
                    reason = f'hold timeout, yaw_err={yaw_text}'
                else:
                    reason = f'hold done, yaw_err={yaw_text}'
                self._finish_cone_exit(reason)
                out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return

            out.linear.x = CONE_EXIT_SPEED
            out.angular.z = self._cone_exit_heading_correction()
            self._debug_cone_log(
                t,
                f'cone EXIT: yaw_err={yaw_text}, heading_src={self._cone_heading_source()}, '
                f'elapsed={elapsed:.2f}s, cmd_v={out.linear.x:.2f}, cmd_w={out.angular.z:+.2f}'
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
        if self.state in ('PARKING_APPROACH', 'PARKING_REPLAY', 'PARKING_DONE'):
            phase = 'PARKING'
        elif self.state in ('CONE_W1', 'CONE_RECOVERY', 'CONE_HEADING_ALIGN', 'CONE_EXIT'):
            phase = 'CONE'
            self._cone_was_active = True
        elif self.state == 'TUNNEL':
            phase = 'TUNNEL'
            self._tunnel_was_active = True
        else:
            if self._cone_was_active:
                self.last_curve_latched = True
                self.last_curve_start_t = self.now_sec()
                self._cone_was_active = False
            if self._tunnel_was_active:
                # 터널 종료 순간 → 필요할 때만 PRE_CAR_FOLLOW 발동
                if PRE_CAR_FOLLOW_SEC > 0.0:
                    self.pre_car_follow_target = self.now_sec() + PRE_CAR_FOLLOW_SEC
                else:
                    self.pre_car_follow_target = None
                self._tunnel_was_active = False

            if self.pre_car_follow_target is not None and \
                    self.now_sec() < self.pre_car_follow_target:
                # PRE_CAR_FOLLOW 창: 설정 시간 동안 무조건 CAR_FOLLOW (앞차 없어도 유지)
                phase = 'CAR_FOLLOW'
            elif self.last_cap is not None:
                # 실제 앞차 추종 → CAR_FOLLOW (PRE_CAR_FOLLOW 창 안팎 동일)
                self.pre_car_follow_target = None
                phase = 'CAR_FOLLOW'
            elif self.last_curve_latched:
                self.pre_car_follow_target = None
                # [테스트용] LAST_CURVE 진입 후 LAST_CURVE_TEST_TIMEOUT_SEC 지나면 강제로 NORMAL 복귀
                if self.now_sec() - self.last_curve_start_t >= LAST_CURVE_TEST_TIMEOUT_SEC:
                    self.last_curve_latched = False
                    self.last_curve_start_t = None
                    # [테스트용] 콘 미션도 다시 트리거 가능하도록 원상 복구
                    self.cone_done = False
                    self.w1_local = None
                    phase = 'NORMAL'
                else:
                    phase = 'LAST_CURVE'
            else:
                # PRE_CAR_FOLLOW 4초 경과 & 앞차 없음 → NORMAL 복귀
                self.pre_car_follow_target = None
                phase = 'NORMAL'

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
