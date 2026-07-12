#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_planner_node.py
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING → CONE_APPROACH → CONE_W1 → CONE_ESCAPE → DRIVING
        DRIVING → TUNNEL → DRIVING
- Cone 미션 (Local Waypoint 기반):
    오도메트리 없이 라이다 상대 좌표만 사용.
    콘이 처음 보이면 CONE_APPROACH: 가장 가까운 중앙콘을 향해 조향(속도는 stanley).
    콘 2개가 AIM_DIST 안에 들어오면 중앙콘/측면콘 오프셋을 기억해
    가상 W1(빈 공간)을 실시간 추종하며 통과.
    W1 도달 후 CONE_ESCAPE: 안쪽 차선(콘이 있던 쪽)으로 강하게 꺾어
    CONE_ESCAPE_SEC 동안 주행한 뒤 DRIVING 복귀 → LAST_CURVE latch.
"""

import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD        = '/stanley/cmd_vel'
SUB_OBSTACLES       = '/obstacles/fused'
SUB_TRAFFIC         = '/traffic_light'
PUB_CMD_VEL         = '/cmd_vel'
PUB_PHASE           = '/behavior/phase'
PUB_WAYPOINT_MARKER = '/behavior/waypoints'   # RViz 웨이포인트 (laser_link 프레임)

# --- 신호등 출발 ---
USE_TRAFFIC_START   = False

# --- 사람 정지 ---
PERSON_STOP_DIST    = 0.8
PERSON_WAIT_SEC     = 2.5
PERSON_PASS_SEC     = 5.0

# --- Car 추종 (속도 캡) ---
CAR_GATE_LAT        = 0.6
CAR_STOP_DIST       = 0.40
CAR_RESUME_DIST     = 0.55
CAR_MID_DIST        = 0.80
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

# --- Cone 탈출 (W1 통과 후 안쪽 차선으로 복귀, 하드코딩) ---
CONE_ESCAPE_SEC     = 1.0     # 강하게 꺾은 채 주행할 시간 [s]
CONE_ESCAPE_W       = 2.5     # 탈출 각속도 크기 [rad/s]
CONE_ESCAPE_SPEED   = 0.7     # 탈출 주행 속도 [m/s]

# --- Tunnel 주행 ---
TUNNEL_ENABLE       = True
TUNNEL_GAIN         = 3.0
TUNNEL_WMAX         = 3.0
TUNNEL_SPEED        = 0.8
TUNNEL_HOLD_SEC     = 1.0
PRE_CAR_FOLLOW_SEC  = 4.0     # 터널 종료 직후 CAR_FOLLOW 유지 시간 [s]

# --- LAST_CURVE (테스트용) ---
LAST_CURVE_TEST_TIMEOUT_SEC = 200.0   # LAST_CURVE 진입 후 이 시간 지나면 강제 NORMAL 복귀 [s]

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

        self.debug             = bool(self.get_parameter('debug').value)
        self.use_traffic_start = bool(self.get_parameter('enable_traffic_light').value)
        self.enable_car_follow = bool(self.get_parameter('enable_car_follow').value)
        self.enable_cone       = bool(self.get_parameter('enable_cone').value)
        self.enable_tunnel     = bool(self.get_parameter('enable_tunnel').value)
        self.enable_person     = bool(self.get_parameter('enable_person').value)

        self.state = 'WAITING_GREEN' if self.use_traffic_start else 'DRIVING'
        self.timer_target = 0.0

        # Car 추종
        self.car_front_dist = None
        self.car_stopped = False
        self.car_log_state = 'none'

        # 신호등
        self.green_seen = False

        # Cone Local Waypoint
        self.cone_done = False
        self.cone_offset_x = 0.0   # 중앙콘 대비 W1의 X 오프셋
        self.cone_offset_y = 0.0   # 중앙콘 대비 W1의 Y 오프셋
        self.w1_local = None       # 차량 기준 W1 현재 좌표 (fx, ly)
        self.cone_target = None    # 접근 중 추종할 중앙콘 좌표 (fx, ly)
        self.cone_escape_dir = 0.0 # W1 통과 후 꺾을 방향 (+1=좌, -1=우)

        # 터널
        self.tunnel_mid_y = 0.0

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

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.phase_pub = self.create_publisher(String, PUB_PHASE, 10)
        self.marker_pub = self.create_publisher(MarkerArray, PUB_WAYPOINT_MARKER, 10)

        self.get_logger().info('Behavior Planner Started (Local Waypoint Mode)')
        self.get_logger().info(
            f'[MISSION] traffic_light={self.use_traffic_start} '
            f'car_follow={self.enable_car_follow} cone={self.enable_cone} '
            f'tunnel={self.enable_tunnel} person={self.enable_person} debug={self.debug}'
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
    #  장애물 콜백 (상태 전이)
    # ============================================================
    def obstacle_callback(self, msg: String):
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

        # 사람 정지 (DRIVING 중에만)
        if not self.enable_person:
            return
        if self.state != 'DRIVING':
            return
        for obs in obstacles:
            if obs.get('class') != PERSON_CLASS:
                continue
            fx = obs.get('forward_x')
            if fx and 0.0 < fx < PERSON_STOP_DIST:
                self.state = 'PERSON_STOP'
                self.timer_target = self.now_sec() + PERSON_WAIT_SEC
                self.get_logger().warn(f'Person {fx:.2f}m → wait')
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
        # W1 도달 -> 안쪽 차선으로 강하게 꺾어 짧게 주행 (하드코딩)
        self.state = 'CONE_ESCAPE'
        self.cone_done = True
        self.w1_local = None
        self.cone_target = None
        self.timer_target = self.now_sec() + CONE_ESCAPE_SEC
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

        # Cone 탈출: 안쪽 차선 쪽으로 고정 조향 후 시간 만료 시 복귀 (→ LAST_CURVE)
        if self.state == 'CONE_ESCAPE':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('✅ Cone escape done → DRIVING')
                self.cmd_pub.publish(self._apply_car_cap(ctrl_msg))
                return
            out.linear.x = CONE_ESCAPE_SPEED
            out.angular.z = self.cone_escape_dir * CONE_ESCAPE_W
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
    #    CONE_APPROACH / CONE_W1 / CONE_ESCAPE -> CONE
    #    TUNNEL -> TUNNEL
    #    그 외 -> PRE_CAR_FOLLOW창 / 앞차 / LAST_CURVE latch / NORMAL
    # ============================================================
    def _publish_phase(self):
        if self.state in ('CONE_APPROACH', 'CONE_W1', 'CONE_ESCAPE'):
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