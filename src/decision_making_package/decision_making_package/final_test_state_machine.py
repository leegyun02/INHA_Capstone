#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_planner_node.py
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING → CONE_APPROACH → CONE_W1 → CONE_ESCAPE → CONE_ESCAPE2 → DRIVING
        DRIVING → TUNNEL → DRIVING
        DRIVING → PARKING_APPROACH → PARKING_REPLAY → PARKING_DONE (종료 상태)
- Cone 미션 (Local Waypoint 기반):
    오도메트리 없이 라이다 상대 좌표만 사용.
    콘이 처음 보이면 CONE_APPROACH: 가장 가까운 중앙콘을 향해 조향(속도는 stanley).
    콘 2개가 AIM_DIST 안에 들어오면 중앙콘/측면콘 오프셋을 기억해
    가상 W1(빈 공간)을 실시간 추종하며 통과.
    W1 도달 후 CONE_ESCAPE: 안쪽 차선(콘이 있던 쪽)으로 강하게 꺾어
    CONE_ESCAPE_SEC 동안 주행.
    이어서 CONE_ESCAPE2: 반대로 한 번 더 꺾어 CONE_ESCAPE2_SEC 동안 유지(S자 정렬)
    한 뒤 DRIVING 복귀 → LAST_CURVE latch.
- 평행주차 미션 (cmd_vel_record_replay_node.py 를 이 상태머신에 통합한 버전):
    my_test_stanley.py 가 가로 정지선(코스 마지막 구간)을 감지하면 /lane/last_lane_detected
    (std_msgs/Bool) 를 True로 발행한다. DRIVING 중 그 신호를 받으면 PARKING_APPROACH로
    전환해 라이다 전방 섹터를 보며 approach_speed로 직진 접근하고, front_trigger_dist
    이내로 들어오면 PARKING_REPLAY로 전환해 PARKING_RECORD_FILE(cmd_vel_record.json)에
    저장된 (dt, vx, wz) 시퀀스를 그대로 재생한다. 재생이 끝나면 PARKING_DONE(정차 유지,
    종료 상태)으로 남는다. 트리거는 노드당 한 번만 발동한다(parking_triggered).
"""

import json
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD        = '/stanley/cmd_vel'
SUB_OBSTACLES       = '/obstacles/fused'
SUB_TRAFFIC         = '/traffic_light'
SUB_SCAN            = '/scan'                     # 주차 자동 접근용 라이다 입력
SUB_LAST_LANE       = '/lane/last_lane_detected'  # my_test_stanley.py 가 발행하는 가로 정지선 감지 결과
PUB_CMD_VEL         = '/cmd_vel'
PUB_PHASE           = '/behavior/phase'
PUB_WAYPOINT_MARKER = '/behavior/waypoints'   # RViz 웨이포인트 (laser_link 프레임)

# --- 신호등 출발 ---
USE_TRAFFIC_START   = False

# --- 사람 정지 ---
PERSON_STOP_DIST    = 0.75
PERSON_WAIT_SEC     = 2.5
PERSON_PASS_SEC     = 5.0

# --- Car 추종 (속도 캡) ---
CAR_GATE_LAT        = 0.6
CAR_STOP_DIST       = 0.25
CAR_RESUME_DIST     = 0.4
CAR_MID_DIST        = 0.60
CAR_CRUISE_DIST     = 1.50
CAR_MID_SPEED       = 0.5
CAR_MAX_CAP         = 1.0

# --- Cone 갈림길 (Local Navigation) ---
CONE_ENABLE         = True
CONE_AIM_DIST       = 1.5     # 콘 2개가 이 안에 들어오면 W1 미션 시작 [m]
CONE_PASS_DIST      = 0.2     # 중앙콘이 이 거리 이내로 들어오면 통과로 간주 [m]
CONE_TIMEOUT_SEC    = 6.0     # 무한루프 방지 타이머 [s]
CONE_AIM_GAIN       = 4.0     # 조향 게인 (angular.z = GAIN * 목표_ly) [1/s] (접근/W1 공용)
CONE_AIM_WMAX       = 3.0     # 각속도 제한 [rad/s] (접근/W1 공용)
CONE_SPEED_MAX      = 0.8     # W1 추종 직진(조향 0) 시 최대 속도 [m/s]
CONE_SPEED_MIN      = 0.5     # W1 추종 최대 조향 시 최소 속도 [m/s]

# --- Cone 1차 탈출 (W1 통과 후 안쪽 차선으로 복귀) ---
#   escape_dir=+1 → 왼쪽 길(LEFT), escape_dir=-1 → 오른쪽 길(RIGHT)
#   왼쪽/오른쪽 갈림길에 따라 파라미터를 따로 튜닝할 수 있음.
# 오른쪽 길 (LEFT)
CONE_ESCAPE_SEC_L    = 1.5     # 강하게 꺾은 채 주행할 시간 [s]
CONE_ESCAPE_W_L      = 2.5     # 탈출 각속도 크기 [rad/s]
CONE_ESCAPE_SPEED_L  = 0.7     # 탈출 주행 속도 [m/s]
# 왼쪽 길 (RIGHT)
CONE_ESCAPE_SEC_R    = 2.0     # 강하게 꺾은 채 주행할 시간 [s]
CONE_ESCAPE_W_R      = 3.0     # 탈출 각속도 크기 [rad/s]
CONE_ESCAPE_SPEED_R  = 0.7     # 탈출 주행 속도 [m/s]

# --- Cone 2차 탈출 (반대로 한번 더 꺾어 유지, S자 정렬) ---
# 오른쪽 길 (LEFT)
CONE_ESCAPE2_SEC_L   = 1.0     # 반대로 꺾은 채 유지할 시간 [s]
CONE_ESCAPE2_W_L     = 2.5     # 반대 탈출 각속도 크기 [rad/s]
CONE_ESCAPE2_SPEED_L = 0.7     # 반대 탈출 주행 속도 [m/s]
# 왼쪽 길 (RIGHT)
CONE_ESCAPE2_SEC_R   = 0.5     # 반대로 꺾은 채 유지할 시간 [s]
CONE_ESCAPE2_W_R     = 2.5     # 반대 탈출 각속도 크기 [rad/s]
CONE_ESCAPE2_SPEED_R = 0.7     # 반대 탈출 주행 속도 [m/s]

# --- Tunnel 주행 ---
TUNNEL_ENABLE       = True
TUNNEL_GAIN         = 3.0
TUNNEL_WMAX         = 3.0
TUNNEL_SPEED        = 0.8
TUNNEL_HOLD_SEC     = 1.0
PRE_CAR_FOLLOW_SEC  = 4.0     # 터널 종료 직후 CAR_FOLLOW 유지 시간 [s]

# --- LAST_CURVE (테스트용) ---
LAST_CURVE_TEST_TIMEOUT_SEC = 200.0   # LAST_CURVE 진입 후 이 시간 지나면 강제 NORMAL 복귀 [s]

# --- 평행주차 (cmd_vel_record_replay_node.py 통합) ---
PARKING_RECORD_FILE            = os.path.expanduser('~/trinity_ws/cmd_vel_record.json')
PARKING_FRONT_ANGLE_CENTER     = 0.0                 # 전방 섹터 중심 각도 (REP103: 0=전방)
PARKING_FRONT_ANGLE_HALF_WIDTH = math.radians(5)     # 전방 섹터 폭
PARKING_FRONT_TRIGGER_DIST     = 0.35   # 전방벽이 이 거리[m] 이하가 되면 자동으로 REPLAY 시작
PARKING_APPROACH_SPEED         = 0.3    # front_trigger_dist 도달 전까지 직진 접근 속도 [m/s]

# --- Class Name ---
PERSON_CLASS        = 'Person'
CAR_CLASS           = 'Car'
CONE_CLASS          = 'Cone'
# ============================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class BehaviorPlannerNode(Node):
    def __init__(self):
        super().__init__('behavior_planner_node')

        # ============================================================
        #  파라미터 (런치/CLI 에서 On/Off)
        #   - debug            : 디버그 로그 출력 여부
        #   - enable_traffic_light : 신호등(초록불 출발) 미션
        #   - enable_car_follow    : 앞차 추종(속도 캡) 미션
        #   - enable_cone          : 콘 갈림길 주행 미션
        #   - enable_tunnel        : 터널 주행 미션
        #   - enable_person        : 사람 정지 미션
        # ============================================================
        self.declare_parameter('debug', False)
        self.declare_parameter('enable_traffic_light', USE_TRAFFIC_START)
        self.declare_parameter('enable_car_follow', True)
        self.declare_parameter('enable_cone', CONE_ENABLE)
        self.declare_parameter('enable_tunnel', TUNNEL_ENABLE)
        self.declare_parameter('enable_person', True)
        self.declare_parameter('enable_parking', True)

        self.debug             = bool(self.get_parameter('debug').value)
        self.use_traffic_start = bool(self.get_parameter('enable_traffic_light').value)
        self.enable_car_follow = bool(self.get_parameter('enable_car_follow').value)
        self.enable_cone       = bool(self.get_parameter('enable_cone').value)
        self.enable_tunnel     = bool(self.get_parameter('enable_tunnel').value)
        self.enable_person     = bool(self.get_parameter('enable_person').value)
        self.enable_parking    = bool(self.get_parameter('enable_parking').value)

        self.state = 'WAITING_GREEN' if self.use_traffic_start else 'DRIVING'
        self.timer_target = 0.0

        # Car 추종
        self.car_front_dist = None
        self.car_stopped = False
        self.car_log_state = 'none'

        # 신호등
        self.green_seen = False

        # 사람 정지 (1회성): 한 번 발동하면 이후로는 사람 무시
        self.person_done = False

        # Cone Local Waypoint
        self.cone_done = False
        self.cone_offset_x = 0.0   # 중앙콘 대비 W1의 X 오프셋
        self.cone_offset_y = 0.0   # 중앙콘 대비 W1의 Y 오프셋
        self.w1_local = None       # 차량 기준 W1 현재 좌표 (fx, ly)
        self.cone_target = None    # 접근 중 추종할 중앙콘 좌표 (fx, ly)
        self.cone_escape_dir = 0.0 # W1 통과 후 꺾을 방향 (+1=좌, -1=우)
        # 탈출 시 방향(좌/우)에 따라 선택되는 활성 파라미터 (_enter_cone_escape에서 세팅)
        self.esc1_sec = CONE_ESCAPE_SEC_L
        self.esc1_w = CONE_ESCAPE_W_L
        self.esc1_speed = CONE_ESCAPE_SPEED_L
        self.esc2_sec = CONE_ESCAPE2_SEC_L
        self.esc2_w = CONE_ESCAPE2_W_L
        self.esc2_speed = CONE_ESCAPE2_SPEED_L

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

        # phase 발행용 latch
        self.last_cap = None
        self._cone_was_active = False
        self._tunnel_was_active = False
        self.last_curve_latched = False
        self.last_curve_start_t = None
        self.pre_car_follow_target = None   # 터널 종료 후 CAR_FOLLOW 만료 시각 (None=비활성)

        self.create_subscription(Twist, SUB_CTRL_CMD, self.cmd_callback, 10)
        self.create_subscription(String, SUB_OBSTACLES, self.obstacle_callback, qos_profile_sensor_data)
        if self.use_traffic_start:
            self.create_subscription(String, SUB_TRAFFIC, self.traffic_callback, 10)
        if self.enable_parking:
            self.create_subscription(LaserScan, SUB_SCAN, self.scan_callback, qos_profile_sensor_data)
            self.create_subscription(Bool, SUB_LAST_LANE, self.last_lane_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.phase_pub = self.create_publisher(String, PUB_PHASE, 10)
        self.marker_pub = self.create_publisher(MarkerArray, PUB_WAYPOINT_MARKER, 10)

        self.get_logger().info('Behavior Planner Started (Local Waypoint Mode)')
        self.get_logger().info(
            f'[MISSION] traffic_light={self.use_traffic_start} '
            f'car_follow={self.enable_car_follow} cone={self.enable_cone} '
            f'tunnel={self.enable_tunnel} person={self.enable_person} '
            f'parking={self.enable_parking} debug={self.debug}'
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def traffic_callback(self, msg: String):
        if self.green_seen:
            return
        if msg.data == 'Green' and self.state == 'WAITING_GREEN':
            self.green_seen = True
            self.state = 'DRIVING'
            self.get_logger().info('🟢 Green Light → Driving')

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

    # ============================================================
    #  장애물 콜백 (상태 전이)
    # ============================================================
    def obstacle_callback(self, msg: String):
        if self.state in ('PARKING_APPROACH', 'PARKING_REPLAY', 'PARKING_DONE'):
            return
        try:
            obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._update_car_follow(obstacles)

        # 터널 진입 (최우선)
        tunnel_active = any(o.get('class') == 'TunnelActive' for o in obstacles)
        if self.enable_tunnel:
            if tunnel_active and self.state != 'TUNNEL':
                self.state = 'TUNNEL'
                self.timer_target = self.now_sec() + TUNNEL_HOLD_SEC
                self.get_logger().warn('🚇 Tunnel detected → TUNNEL')
            if self.state == 'TUNNEL':
                self._update_tunnel_mid(obstacles)
                return

        # Cone 미션
        if self.enable_cone and not self.cone_done:
            if self.state == 'DRIVING':
                # 콘이 처음 보이는 순간(거리 무관) -> 접근 시작
                self._trigger_cone_approach(obstacles)
            elif self.state == 'CONE_APPROACH':
                # 매 프레임 중앙콘 추종 좌표 갱신 + 2개가 AIM_DIST 들어오면 W1 시작
                self._update_cone_approach(obstacles)
                self._trigger_cone_mission(obstacles)
            elif self.state == 'CONE_W1':
                self._update_cone_w1(obstacles)

        # 사람 정지 (DRIVING 중에만, 코스당 1회만 발동)
        if not self.enable_person:
            return
        if self.person_done:
            return
        if self.state != 'DRIVING':
            return
        # 앞차 추종(CAR_FOLLOW) 중이면 사람 감지해도 정지하지 않음
        if self.car_front_dist is not None:
            return
        for obs in obstacles:
            if obs.get('class') != PERSON_CLASS:
                continue
            fx = obs.get('forward_x')
            if fx and 0.0 < fx < PERSON_STOP_DIST:
                self.state = 'PERSON_STOP'
                self.person_done = True
                self.timer_target = self.now_sec() + PERSON_WAIT_SEC
                self.get_logger().warn(f'Person {fx:.2f}m → wait (1회성, 이후 무시)')
                break

    def _update_tunnel_mid(self, obstacles):
        left_ly, right_ly = None, None
        for o in obstacles:
            if o.get('class') != 'TunnelWall':
                continue
            if o.get('side') == 'left':
                left_ly = o.get('lateral_y')
            elif o.get('side') == 'right':
                right_ly = o.get('lateral_y')

        if left_ly is not None and right_ly is not None:
            self.tunnel_mid_y = (left_ly + right_ly) / 2.0
        elif left_ly is not None:
            self.tunnel_mid_y = left_ly - 0.3
        elif right_ly is not None:
            self.tunnel_mid_y = right_ly + 0.3

    def _trigger_cone_approach(self, obstacles):
        # 콘이 하나라도 보이면 CONE_APPROACH 진입 (거리 조건 없음)
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS:
                continue
            fx = obs.get('forward_x')
            if fx is not None and fx > 0.0:
                self.state = 'CONE_APPROACH'
                self.get_logger().warn('🚧 Cone seen → CONE_APPROACH (aim center cone)')
                return

    def _update_cone_approach(self, obstacles):
        # 접근 중: 가장 가까운 콘(중앙콘)을 추종 목표로 갱신
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None and fx > 0.0:
                cones.append((math.hypot(fx, ly), fx, ly))

        if not cones:
            self.cone_target = None
            return
        cones.sort(key=lambda c: c[0])
        _, cx, cy = cones[0]
        self.cone_target = (cx, cy)

    def _trigger_cone_mission(self, obstacles):
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None and 0.0 < fx <= CONE_AIM_DIST:
                cones.append((math.hypot(fx, ly), fx, ly))

        if len(cones) < 2:
            return

        # 거리순 정렬: [0]=중앙콘(가까움), [1]=측면콘
        cones.sort(key=lambda c: c[0])
        _, cx, cy = cones[0]
        _, sx, sy = cones[1]

        # 측면콘 정반대(빈 공간) 방향 오프셋 기억: W1 = 2C - S = C + (C - S)
        self.cone_offset_x = cx - sx
        self.cone_offset_y = cy - sy
        self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)

        # W1은 측면콘 반대편에 생기므로, 안쪽(콘 쪽) 차선은 offset_y의 반대 방향
        self.cone_escape_dir = -1.0 if self.cone_offset_y > 0.0 else 1.0

        self.state = 'CONE_W1'
        self.timer_target = self.now_sec() + CONE_TIMEOUT_SEC
        side = 'LEFT' if self.cone_escape_dir > 0 else 'RIGHT'
        self.get_logger().warn(f'🚧 Cone Trigger! Offset locked → Aiming W1 (escape={side})')

    def _update_cone_w1(self, obstacles):
        # 매 프레임 중앙콘 위치를 다시 찾아 W1 갱신
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None:
                cones.append((math.hypot(fx, ly), fx, ly))

        if not cones:
            # 콘이 시야에서 사라짐 -> 통과로 간주
            self._enter_cone_escape('Cones lost (passed)')
            return

        cones.sort(key=lambda c: c[0])
        _, cx, cy = cones[0]

        if cx < CONE_PASS_DIST:
            self._enter_cone_escape('Passed Center Cone')
        else:
            self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)

    def _enter_cone_escape(self, reason: str):
        # W1 도달 -> 안쪽 차선으로 강하게 꺾어 짧게 주행 (1차 탈출)
        self.state = 'CONE_ESCAPE'
        self.cone_done = True
        self.w1_local = None
        self.cone_target = None
        # 방향(좌/우)에 따라 1차/2차 탈출 파라미터 세팅
        if self.cone_escape_dir > 0:  # 왼쪽 길 (LEFT)
            self.esc1_sec, self.esc1_w, self.esc1_speed = \
                CONE_ESCAPE_SEC_L, CONE_ESCAPE_W_L, CONE_ESCAPE_SPEED_L
            self.esc2_sec, self.esc2_w, self.esc2_speed = \
                CONE_ESCAPE2_SEC_L, CONE_ESCAPE2_W_L, CONE_ESCAPE2_SPEED_L
        else:  # 오른쪽 길 (RIGHT)
            self.esc1_sec, self.esc1_w, self.esc1_speed = \
                CONE_ESCAPE_SEC_R, CONE_ESCAPE_W_R, CONE_ESCAPE_SPEED_R
            self.esc2_sec, self.esc2_w, self.esc2_speed = \
                CONE_ESCAPE2_SEC_R, CONE_ESCAPE2_W_R, CONE_ESCAPE2_SPEED_R
        self.timer_target = self.now_sec() + self.esc1_sec
        side = 'LEFT' if self.cone_escape_dir > 0 else 'RIGHT'
        self.get_logger().warn(f'✅ {reason} → CONE_ESCAPE (turn {side})')

    # ============================================================
    #  Car 추종 속도 캡
    # ============================================================
    def _update_car_follow(self, obstacles):
        if not self.enable_car_follow:
            self.car_front_dist = None
            return
        nearest = None
        for obs in obstacles:
            if obs.get('class') != CAR_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is None or ly is None:
                continue
            if fx <= 0.0 or abs(ly) > CAR_GATE_LAT:
                continue
            if nearest is None or fx < nearest:
                nearest = fx
        self.car_front_dist = nearest

    def _car_speed_cap(self):
        d = self.car_front_dist
        if d is None:
            self.car_stopped = False
            return None

        if self.car_stopped:
            if d > CAR_RESUME_DIST:
                self.car_stopped = False
            else:
                return 0.0
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

    def _apply_car_cap(self, cmd: Twist):
        cap = self._car_speed_cap()
        self.last_cap = cap
        if cap is None:
            self.car_log_state = 'none'
        elif cap == 0.0:
            self.car_log_state = 'stop'
        else:
            self.car_log_state = 'cap'

        if cap is not None and cmd.linear.x > cap:
            if cmd.linear.x > 1e-3:
                cmd.angular.z *= (cap / cmd.linear.x)
            cmd.linear.x = cap
        return cmd

    # ============================================================
    #  RViz 마커
    # ============================================================
    def _publish_waypoint_markers(self):
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.w1_local is None or self.state != 'CONE_W1':
            self.marker_pub.publish(marker_array)
            return

        m1 = Marker()
        m1.header.frame_id = 'laser_link'
        m1.header.stamp = self.get_clock().now().to_msg()
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

    # ============================================================
    #  제어 콜백 (상태별 게이팅 후 /cmd_vel 발행)
    # ============================================================
    def cmd_callback(self, ctrl_msg: Twist):
        t = self.now_sec()
        out = Twist()

        if self.enable_parking and self.state == 'DRIVING' and self.last_lane_detected \
                and self.last_curve_latched and not self.parking_triggered:
            self._start_parking_approach()

        self._publish_phase()
        self._publish_waypoint_markers()

        if self.debug:
            self.get_logger().info(
                f'[DEBUG] state={self.state} in(v={ctrl_msg.linear.x:.2f}, '
                f'w={ctrl_msg.angular.z:.2f}) car_d={self.car_front_dist} cap={self.last_cap}',
                throttle_duration_sec=1.0,
            )

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
                self.get_logger().info('🚇 Tunnel end → lane following')
                self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))
                return
            out.linear.x = TUNNEL_SPEED
            out.angular.z = clamp(TUNNEL_GAIN * self.tunnel_mid_y, -TUNNEL_WMAX, TUNNEL_WMAX)
            self.cmd_pub.publish(self._apply_car_cap(out))
            return

        # Cone 접근: 중앙콘을 향해 조향, 속도는 stanley 그대로 (+car_cap)
        if self.state == 'CONE_APPROACH':
            if self.cone_target is not None:
                out.angular.z = clamp(
                    CONE_AIM_GAIN * self.cone_target[1], -CONE_AIM_WMAX, CONE_AIM_WMAX
                )
            else:
                out.angular.z = 0.0
            out.linear.x = ctrl_msg.linear.x
            self.cmd_pub.publish(self._apply_car_cap(out))
            return

        # Cone 미션: W1 추종
        if self.state == 'CONE_W1':
            if t > self.timer_target:
                self.state = 'DRIVING'
                self.cone_done = True
                self.get_logger().warn('🚧 Cone Mission Timeout!')
                self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))
                return

            if self.w1_local is not None:
                # W1의 가상 측면 오차(ly)를 향해 조향, 조향 클수록 감속
                w = clamp(CONE_AIM_GAIN * self.w1_local[1], -CONE_AIM_WMAX, CONE_AIM_WMAX)
                steer_ratio = abs(w) / CONE_AIM_WMAX
                out.linear.x = CONE_SPEED_MAX - steer_ratio * (CONE_SPEED_MAX - CONE_SPEED_MIN)
                out.angular.z = w
            else:
                out.linear.x = CONE_SPEED_MAX
                out.angular.z = 0.0
            self.cmd_pub.publish(out)
            return

        # Cone 1차 탈출: 안쪽 차선 쪽으로 고정 조향 -> 만료 시 2차 탈출로 전환
        if self.state == 'CONE_ESCAPE':
            if t >= self.timer_target:
                self.state = 'CONE_ESCAPE2'
                self.timer_target = t + self.esc2_sec
                side2 = 'LEFT' if self.cone_escape_dir < 0 else 'RIGHT'
                self.get_logger().info(f'✅ Cone escape1 done → CONE_ESCAPE2 (turn {side2})')
                # 전환 즉시 2차 탈출 명령 발행 (아래 블록으로 fall-through)
            else:
                out.linear.x = self.esc1_speed
                out.angular.z = self.cone_escape_dir * self.esc1_w
                self.cmd_pub.publish(out)
                return

        # Cone 2차 탈출: 반대로 꺾어 유지 -> 만료 시 DRIVING 복귀 (→ LAST_CURVE)
        if self.state == 'CONE_ESCAPE2':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('✅ Cone escape2 done → DRIVING')
                self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))
                return
            out.linear.x = self.esc2_speed
            out.angular.z = -self.cone_escape_dir * self.esc2_w
            self.cmd_pub.publish(out)
            return

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
            self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))
            return

        # 일반 주행 (Stanley 기반)
        self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))

    # ============================================================
    #  phase 발행
    #    CONE_APPROACH / CONE_W1 / CONE_ESCAPE / CONE_ESCAPE2 -> CONE
    #    TUNNEL -> TUNNEL
    #    그 외 -> PRE_CAR_FOLLOW창 / 앞차 / LAST_CURVE latch / NORMAL
    # ============================================================
    def _publish_phase(self):
        if self.state in ('PARKING_APPROACH', 'PARKING_REPLAY', 'PARKING_DONE'):
            phase = 'PARKING'
        elif self.state in ('CONE_APPROACH', 'CONE_W1', 'CONE_ESCAPE', 'CONE_ESCAPE2'):
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
                # 터널 종료 순간 -> PRE_CAR_FOLLOW 발동 (CAR_FOLLOW 동일 동작)
                self.pre_car_follow_target = self.now_sec() + PRE_CAR_FOLLOW_SEC
                self._tunnel_was_active = False

            if self.pre_car_follow_target is not None and self.now_sec() < self.pre_car_follow_target:
                # PRE_CAR_FOLLOW 창: 앞차 없어도 CAR_FOLLOW 유지
                phase = 'CAR_FOLLOW'
            elif self.last_cap is not None:
                # 실제 앞차 추종
                self.pre_car_follow_target = None
                phase = 'CAR_FOLLOW'
            elif self.last_curve_latched:
                self.pre_car_follow_target = None
                # [테스트용] 일정 시간 후 강제 NORMAL 복귀 + 콘 미션 재무장
                if self.now_sec() - self.last_curve_start_t >= LAST_CURVE_TEST_TIMEOUT_SEC:
                    self.last_curve_latched = False
                    self.last_curve_start_t = None
                    self.cone_done = False
                    self.w1_local = None
                    phase = 'NORMAL'
                else:
                    phase = 'LAST_CURVE'
            else:
                self.pre_car_follow_target = None
                phase = 'NORMAL'

        msg = String()
        msg.data = phase
        self.phase_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()