#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_parking_node.py
- 라이다(/scan) 기반 평행주차 전용 독립 노드 (추후 behavior_planner에 병합 예정).
- 주차공간: 진입로만 뚫려있고 나머지 3면은 벽 (전방벽 / 우측벽 / 안쪽벽).

상태 흐름:
    IDLE
      -> (trigger 수신 또는 AUTO_START) ALIGN
    ALIGN   : 우측벽과의 거리를 RIGHT_TARGET_DIST로 유지하며 전진,
              전방벽이 FRONT_STOP_DIST 이내로 들어오면 정지 트리거.
              (셔틀 왕복은 시작 자세가 항상 동일해야 하므로,
               여기서 라이다로 위치를 맞춘 뒤에만 아래 시퀀스를 시작한다.)
    ALIGN_HOLD : 완전 정지될 때까지 잠시 대기.
    SHUTTLE : 후진↔전진을 반복하며 주차공간 안쪽으로 파고드는 단계.
              - 진입 전(안쪽벽 미검출): 후방(뒤쪽, 180°) 섹터에 아무것도 안
                잡히면(진입로가 뚫려있어 range 밖) 오실레이션 없이 그냥
                SHUTTLE_REV_ANGLE 로 후진만 계속한다. 전방벽 기준으로 판단하면
                후진할수록 전방벽과는 계속 멀어지기만 해서, 아직 주차구역에
                들어가지도 않았는데 왕복 로직이 오작동하기 때문에 후방벽 기준으로 바꿈.
              - 후방 섹터에 안쪽벽이 처음 감지되는 순간부터 그 거리를 기준으로
                REV/FWD 왕복 시작:
                REV(후진): 안쪽벽 거리가 SHUTTLE_REAR_NEAR_DIST 이하로
                가까워지면 FWD로 전환.
                FWD(전진): 안쪽벽 거리가 SHUTTLE_REAR_FAR_DIST 이상으로
                멀어지면 REV로 전환 (1 사이클 증가).
              - 매 tick마다 좌측벽 거리(LEFT_TARGET_DIST±LEFT_TOL)와
                안쪽벽 거리(REAR_TARGET_DIST±REAR_TOL)가 동시에 만족되면 즉시 종료.
              - SHUTTLE_MAX_CYCLES / SHUTTLE_MAX_SEC 를 넘기면 안전 타임아웃으로 강제 종료.
    DONE    : 정지 유지.

- 각도/시간/속도/거리는 전부 하드코딩 튜닝 파라미터이며 실차에서 맞추는 것을 전제로 함.
- 좌표계는 REP103 기준(x=전방, y=좌측, z=위)을 가정: 각도 0=전방, +90deg=좌측, -90deg=우측, ±180deg=후방.
  실제 라이다 장착 방향이 다르면 FRONT/RIGHT/LEFT/REAR 각도 및 조향 부호를 확인해서 맞출 것.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_SCAN            = '/scan'
SUB_TRIGGER         = '/parking/start'   # std_msgs/Bool: True 수신 시 ALIGN 시작 (병합 전 단독 테스트용)
PUB_CMD_VEL         = '/cmd_vel'
PUB_PHASE           = '/behavior/phase'

# --- 라이다 섹터 각도 (REP103: 전방=0, 좌측=+90deg, 우측=-90deg, 후방=180deg) ---
FRONT_ANGLE_CENTER      = 0.0
FRONT_ANGLE_HALF_WIDTH  = math.radians(5)
RIGHT_ANGLE_CENTER      = -math.pi / 2.0
RIGHT_ANGLE_HALF_WIDTH  = math.radians(10)
LEFT_ANGLE_CENTER       = math.pi / 2.0
LEFT_ANGLE_HALF_WIDTH   = math.radians(10)
REAR_ANGLE_CENTER       = math.pi
REAR_ANGLE_HALF_WIDTH   = math.radians(10)

# --- 시작 자세 정렬 (ALIGN) ---
AUTO_START          = True    # True면 트리거 없이 노드 시작과 동시에 ALIGN 진행 (단독 테스트용)
ALIGN_SPEED         = 0.4     # 정렬 중 전진 속도 [m/s]
ALIGN_STEER_GAIN    = 0.5     # 우측벽 거리 오차 -> 조향 게인 [1/s] (부호는 실차에서 확인)
ALIGN_WMAX          = 1.0     # 정렬 중 최대 각속도 [rad/s]
RIGHT_TARGET_DIST   = 0.30    # 정렬 목표: 우측벽과의 거리 [m]
FRONT_STOP_DIST     = 0.40    # 전방벽이 이 거리 이내면 정지 후 주차 시퀀스 시작 [m]
ALIGN_HOLD_SEC      = 0.5     # 정지 트리거 후 완전 정지까지 대기 시간 [s]

# --- 주차 셔틀 왕복 (전부 하드코딩 각도/속도 + 거리 트리거, 실차에서 튜닝) ---
SHUTTLE_REV_ANGLE   = -45.0   # 후진(REV) 중 조향 각속도 (angular.z 그대로 사용) [rad/s]
SHUTTLE_REV_SPEED   = 0.2     # 후진 속도 [m/s] (양수로 입력, 내부에서 음수로 적용)
SHUTTLE_FWD_ANGLE   = 45.0    # 전진(FWD) 중 조향 각속도 (보통 REV의 반대 부호) [rad/s]
SHUTTLE_FWD_SPEED   = 0.2     # 전진 속도 [m/s]

SHUTTLE_REAR_NEAR_DIST = 0.30  # 후진 중 안쪽벽이 이 거리 이하로 가까워지면 FWD로 전환 [m]
SHUTTLE_REAR_FAR_DIST  = 0.60  # 전진 중 안쪽벽이 이 거리 이상으로 멀어지면 REV로 전환 [m]

LEFT_TARGET_DIST    = 0.30    # 주차 종료 목표: 좌측벽과의 거리 [m]
LEFT_TOL            = 0.05    # 좌측벽 목표 허용 오차 [m]
REAR_TARGET_DIST    = 0.30    # 주차 종료 목표: 안쪽벽과의 거리 [m]
REAR_TOL            = 0.05    # 안쪽벽 목표 허용 오차 [m]

SHUTTLE_MAX_CYCLES  = 6       # 안전장치: 최대 왕복 횟수 (초과 시 강제 종료)
SHUTTLE_MAX_SEC     = 30.0    # 안전장치: 셔틀 진입 후 최대 허용 시간 [s] (초과 시 강제 종료)

# --- 제어 주기 ---
CONTROL_HZ          = 20.0
LOG_THROTTLE_SEC     = 1.0
# ============================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ParallelParkingNode(Node):
    def __init__(self):
        super().__init__('parallel_parking_node')

        self.state = 'IDLE'
        self.timer_target = 0.0
        self.latest_scan = None

        # SHUTTLE 왕복 상태
        self.shuttle_dir = 'REV'
        self.shuttle_cycles = 0
        self.shuttle_start_t = 0.0
        self.shuttle_rear_seen = False   # 후방(안쪽벽)이 처음 감지됐는지

        self.create_subscription(LaserScan, SUB_SCAN, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Bool, SUB_TRIGGER, self.trigger_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.phase_pub = self.create_publisher(String, PUB_PHASE, 10)

        self.create_timer(1.0 / CONTROL_HZ, self.control_loop)

        self.get_logger().info(
            f'Parallel Parking Node Started (auto_start={AUTO_START}, '
            f'right_target={RIGHT_TARGET_DIST}m, front_stop={FRONT_STOP_DIST}m)'
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ============================================================
    #  구독 콜백
    # ============================================================
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def trigger_callback(self, msg: Bool):
        if msg.data and self.state == 'IDLE':
            self.state = 'ALIGN'
            self.get_logger().info('▶ Parking triggered → ALIGN')

    # ============================================================
    #  라이다 섹터 거리 (지정 각도 주변 range 값들의 중앙값)
    # ============================================================
    def _sector_distance(self, scan: LaserScan, center_rad: float, half_width_rad: float):
        dists = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                if abs(normalize_angle(angle - center_rad)) <= half_width_rad:
                    dists.append(r)
            angle += scan.angle_increment
        if not dists:
            return None
        dists.sort()
        return dists[len(dists) // 2]

    def _sector_dists(self):
        scan = self.latest_scan
        if scan is None:
            return None, None, None, None
        front = self._sector_distance(scan, FRONT_ANGLE_CENTER, FRONT_ANGLE_HALF_WIDTH)
        right = self._sector_distance(scan, RIGHT_ANGLE_CENTER, RIGHT_ANGLE_HALF_WIDTH)
        left = self._sector_distance(scan, LEFT_ANGLE_CENTER, LEFT_ANGLE_HALF_WIDTH)
        rear = self._sector_distance(scan, REAR_ANGLE_CENTER, REAR_ANGLE_HALF_WIDTH)
        return front, right, left, rear

    # ============================================================
    #  메인 제어 루프
    # ============================================================
    def control_loop(self):
        t = self.now_sec()
        out = Twist()

        self._publish_phase()

        if self.state == 'IDLE':
            if AUTO_START:
                self.state = 'ALIGN'
                self.get_logger().info('▶ Auto start → ALIGN')
            self.cmd_pub.publish(out)
            return

        if self.state == 'ALIGN':
            front, right, _left, _rear = self._sector_dists()
            if front is None:
                self.get_logger().warn('ALIGN: no /scan yet, holding', throttle_duration_sec=LOG_THROTTLE_SEC)
                self.cmd_pub.publish(out)
                return

            if front <= FRONT_STOP_DIST:
                self.state = 'ALIGN_HOLD'
                self.timer_target = t + ALIGN_HOLD_SEC
                self.get_logger().info(
                    f'✅ Front wall {front:.2f}m reached → ALIGN_HOLD'
                )
                self.cmd_pub.publish(Twist())
                return

            steer = 0.0
            if right is not None:
                steer = clamp(-ALIGN_STEER_GAIN * (right - RIGHT_TARGET_DIST), -ALIGN_WMAX, ALIGN_WMAX)

            out.linear.x = ALIGN_SPEED
            out.angular.z = steer
            self.get_logger().info(
                f'ALIGN: front={front:.2f}m right={"n/a" if right is None else f"{right:.2f}m"} '
                f'steer={steer:+.2f}',
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )
            self.cmd_pub.publish(out)
            return

        if self.state == 'ALIGN_HOLD':
            self.cmd_pub.publish(Twist())
            if t >= self.timer_target:
                self.state = 'SHUTTLE'
                self.shuttle_dir = 'REV'
                self.shuttle_cycles = 0
                self.shuttle_start_t = t
                self.shuttle_rear_seen = False
                self.get_logger().info('▶ SHUTTLE (안쪽벽 보일 때까지 후진 후 왕복)')
            return

        if self.state == 'SHUTTLE':
            front, right, left, rear = self._sector_dists()
            if front is None:
                self.get_logger().warn('SHUTTLE: no /scan yet, holding', throttle_duration_sec=LOG_THROTTLE_SEC)
                self.cmd_pub.publish(Twist())
                return

            if (t - self.shuttle_start_t) >= SHUTTLE_MAX_SEC:
                self.state = 'DONE'
                self.get_logger().warn('⚠️ SHUTTLE 목표 미도달 → 안전 타임아웃 DONE')
                self.cmd_pub.publish(Twist())
                return

            # 안쪽벽이 아직 후방 섹터에 안 잡히면(주차구역 진입 전) 오실레이션 없이 계속 후진만
            if rear is None:
                out.linear.x = -SHUTTLE_REV_SPEED
                out.angular.z = SHUTTLE_REV_ANGLE
                self.get_logger().info(
                    'SHUTTLE[ENTER]: rear=n/a (안쪽벽 미검출) → 후진 계속',
                    throttle_duration_sec=LOG_THROTTLE_SEC,
                )
                self.cmd_pub.publish(out)
                return

            if not self.shuttle_rear_seen:
                self.shuttle_rear_seen = True
                self.shuttle_dir = 'REV'
                self.get_logger().info(f'👀 안쪽벽 감지 (rear={rear:.2f}m) → 왕복 시작')

            rear_ok = abs(rear - REAR_TARGET_DIST) <= REAR_TOL
            left_ok = left is not None and abs(left - LEFT_TARGET_DIST) <= LEFT_TOL
            if rear_ok and left_ok:
                self.state = 'DONE'
                self.get_logger().info(
                    f'🅿️ 목표 도달 (rear={rear:.2f}m, left={left:.2f}m) → DONE'
                )
                self.cmd_pub.publish(Twist())
                return

            if self.shuttle_cycles >= SHUTTLE_MAX_CYCLES:
                self.state = 'DONE'
                self.get_logger().warn(
                    f'⚠️ SHUTTLE 목표 미도달 (rear={rear:.2f}m, '
                    f'left={"n/a" if left is None else f"{left:.2f}m"}) → 최대 반복 DONE'
                )
                self.cmd_pub.publish(Twist())
                return

            if self.shuttle_dir == 'REV':
                out.linear.x = -SHUTTLE_REV_SPEED
                out.angular.z = SHUTTLE_REV_ANGLE
                if rear <= SHUTTLE_REAR_NEAR_DIST:
                    self.shuttle_dir = 'FWD'
                    self.get_logger().info(f'↔ rear={rear:.2f}m ≤ near → FWD')
            else:
                out.linear.x = SHUTTLE_FWD_SPEED
                out.angular.z = SHUTTLE_FWD_ANGLE
                if rear >= SHUTTLE_REAR_FAR_DIST:
                    self.shuttle_dir = 'REV'
                    self.shuttle_cycles += 1
                    self.get_logger().info(f'↔ rear={rear:.2f}m ≥ far → REV (cycle {self.shuttle_cycles})')

            self.get_logger().info(
                f'SHUTTLE[{self.shuttle_dir}]: rear={rear:.2f}m '
                f'left={"n/a" if left is None else f"{left:.2f}m"} cycle={self.shuttle_cycles}',
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )
            self.cmd_pub.publish(out)
            return

        # DONE
        self.cmd_pub.publish(Twist())

    def _publish_phase(self):
        if self.state == 'IDLE':
            phase = 'NORMAL'
        elif self.state == 'DONE':
            phase = 'PARKING_DONE'
        else:
            phase = 'PARKING'
        msg = String()
        msg.data = phase
        self.phase_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParallelParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
