#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cmd_vel_record_replay_node.py
- 조이스틱 등으로 수동 조작한 주행을 그대로 녹화했다가, 트리거를 주면 녹화된 그대로
  재생하는 범용 노드 (특정 미션에 종속되지 않음. 주차 등 다른 노드에서도 재사용 가능).

- 녹화 소스는 record_source 파라미터로 선택한다 (기본값 'cmd_vel'):
    'cmd_vel' : /cmd_vel(geometry_msgs/Twist) 를 그대로 녹화.
                노트북에서 ros-humble-teleop-twist-keyboard 같은 걸로 직접
                /cmd_vel 을 발행하며 조작할 때 쓴다 (실제 명령값 그대로라 가장 정확).
    'odom'    : /odom(nav_msgs/Odometry, twist.twist.linear.x/angular.z) 을 녹화.
                이 LIMO는 조이스틱이 RC 수신기를 통해 베이스 펌웨어에 직접 꽂혀서
                /cmd_vel 을 거치지 않기 때문에(limo_driver.cpp 의 control_mode 참고),
                조이스틱으로 조작할 때는 /cmd_vel 대신 베이스 실제 휠 피드백인
                /odom 을 녹화 소스로 써야 한다.
- 재생은 항상 /cmd_vel(geometry_msgs/Twist) 로 발행한다. record_source='odom'으로
  녹화했다면, 재생이 실제로 먹히려면 그 시점에 로봇이 "명령 모드"여야 한다 (조이스틱
  리모컨에 모드 스위치가 있으면 전환 필요 — 하드웨어 스위치라 코드로 해결 불가).
  record_source='cmd_vel'(키보드 등)이면 이런 모드 전환 문제 없음.
- 녹화되는 dt(각 명령의 유지시간)는 실제로 키를 얼마나 오래 눌렀는지를 재지 않고,
  항상 record_fixed_dt 파라미터 값(기본 0.5초)으로 고정한다. "키 한 번 눌렀다 뗐다"를
  전부 같은 크기의 단위 이동으로 취급하는 것 — 그래야 같은 파일을 다시 재생할 때마다
  항상 같은 만큼(dt * vx/wz) 움직인다는 게 보장된다 (사람이 누른 실제 텀은 들쑥날쑥해서
  그대로 쓰면 재생 결과가 매번 달라짐).
- 자동 트리거(auto_trigger 파라미터, 기본 True): IDLE 상태에서 이 노드가 직접
  approach_speed(기본 0.3m/s)로 직진하며 /scan 전방 섹터 거리를 계속 보다가,
  front_trigger_dist(기본 0.35m) 이하가 되면 그 자리에서 자동으로 REPLAY를 시작한다.
  즉 노드를 켜두기만 하면 전방벽까지 직진 접근 → 도달 시 녹화된 주차 시퀀스 재생까지
  전부 이 노드 혼자 처리한다 (다른 노드가 별도로 전진시켜줄 필요 없음). JSON 재생이
  끝나면 그게 주차 완료. 노드 하나당 한 번만 자동 발동한다(재발동 방지). 수동으로도
  여전히 /cmd_vel_record/replay 로 트리거 가능.
- require_last_lane_trigger 파라미터(기본 True): True면 my_test_stanley.py 가 발행하는
  /lane/last_lane_detected(std_msgs/Bool) 가 True로 들어오기 전까지는 접근 주행 자체를
  시작하지 않고 정지 대기한다. 차선 추종 노드와 같이 켜서 "코스 마지막 구간(가로
  정지선) 인식 → 그때부터 파킹 노드가 접근+재생"까지 자동으로 이어지게 하기 위함.
  파킹 로직만 단독으로 테스트하고 싶으면 false 로 끄면 이전처럼 노드 시작과 동시에
  바로 접근 주행을 시작한다.

상태: IDLE -> RECORDING -> IDLE(저장) -> REPLAYING -> IDLE

사용법 (키보드로 조작할 경우, 터미널 3개):
    ros2 run decision_making_package cmd_vel_record_replay_node
    ros2 run teleop_twist_keyboard teleop_twist_keyboard   # /cmd_vel 로 직접 조작
    ros2 topic pub -1 /cmd_vel_record/start std_msgs/msg/Bool "{data: true}"
    (키보드로 원하는 동작 조작; teleop_twist_keyboard 는 키를 뗘도 마지막 속도가 유지되니
     정지하려면 k 또는 스페이스바로 확실히 0으로 만들고 stop 트리거를 줄 것)
    ros2 topic pub -1 /cmd_vel_record/stop  std_msgs/msg/Bool "{data: true}"
    (teleop_twist_keyboard 종료 후, 재생 시 cmd_vel 발행자가 이 노드 하나만 되게 할 것)
    ros2 topic pub -1 /cmd_vel_record/replay std_msgs/msg/Bool "{data: true}"
    ros2 topic pub -1 /cmd_vel_record/cancel std_msgs/msg/Bool "{data: true}"   # 재생 중 긴급 취소

주의:
- REPLAYING 중에는 이 노드가 /cmd_vel 을 직접 발행한다. 다른 노드가 동시에 같은
  /cmd_vel 을 발행하고 있으면 두 발행이 충돌하니 하나만 켜져 있어야 한다.
- 녹화 파일은 JSON(list of [dt, linear_x, angular_z])으로 record_file 경로에 저장된다.
"""

import json
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


# ============================================================
#                  튜닝 파라미터
# ============================================================
SUB_CMD_VEL_IN   = '/cmd_vel'                 # 녹화 소스(record_source='cmd_vel'): 실제 발행된 명령 그대로
SUB_ODOM_IN      = '/odom'                    # 녹화 소스(record_source='odom'): 베이스 실제 휠 피드백
SUB_RECORD_START = '/cmd_vel_record/start'   # std_msgs/Bool(true): 녹화 시작
SUB_RECORD_STOP  = '/cmd_vel_record/stop'    # std_msgs/Bool(true): 녹화 종료 + 파일 저장
SUB_REPLAY_START = '/cmd_vel_record/replay'  # std_msgs/Bool(true): 저장된 기록 재생 시작
SUB_REPLAY_CANCEL = '/cmd_vel_record/cancel' # std_msgs/Bool(true): 재생/녹화 강제 취소
PUB_CMD_VEL      = '/cmd_vel'                # 재생 중에만 발행
PUB_STATE        = '/cmd_vel_record/state'   # 현재 상태 문자열 (IDLE/RECORDING/REPLAYING)

DEFAULT_RECORD_FILE = os.path.expanduser('~/trinity_ws/cmd_vel_record.json')

SUB_SCAN         = '/scan'                    # 자동 트리거용 라이다 입력
SUB_LAST_LANE    = '/lane/last_lane_detected'  # my_test_stanley.py 가 발행하는 가로 정지선 감지 결과
FRONT_ANGLE_CENTER     = 0.0                  # 전방 섹터 중심 각도 (REP103: 0=전방)
FRONT_ANGLE_HALF_WIDTH = math.radians(5)      # 전방 섹터 폭
FRONT_TRIGGER_DIST = 0.35   # 전방벽이 이 거리[m] 이하가 되면 자동으로 REPLAY 시작
APPROACH_SPEED   = 0.3    # front_trigger_dist 에 도달하기 전까지 직진 접근 속도 [m/s]

CONTROL_HZ       = 20.0   # 재생 제어 주기
RECORD_FIXED_DT   = 0.5    # 키 한 번(=/cmd_vel 메시지 하나)당 재생 시 유지할 고정 시간 [s]
                            # 실제로 그 키를 얼마나 오래 눌렀는지(밀리초 단위 텀)는 재지 않고,
                            # "한 번 눌렀다 뗐다"를 전부 이 값으로 통일한다. 그래야 같은 파일을
                            # 다시 재생했을 때 항상 같은 만큼(dt * vx/wz) 움직인다는 게 보장된다.
# ============================================================


class CmdVelRecordReplayNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_record_replay_node')

        self.declare_parameter('record_file', DEFAULT_RECORD_FILE)
        self.record_file = self.get_parameter('record_file').value

        self.declare_parameter('record_source', 'cmd_vel')  # 'cmd_vel' or 'odom'
        self.record_source = self.get_parameter('record_source').value

        self.declare_parameter('replay_speed_scale', 1.0)  # 1.0=녹화 그대로, 2.0=같은 경로 2배 빠르게, 0.5=절반 느리게 (시간+속도 동시 스케일)
        self.replay_speed_scale = float(self.get_parameter('replay_speed_scale').value)

        self.declare_parameter('record_fixed_dt', RECORD_FIXED_DT)
        self.record_fixed_dt = float(self.get_parameter('record_fixed_dt').value)

        self.declare_parameter('auto_trigger', True)
        self.auto_trigger = bool(self.get_parameter('auto_trigger').value)

        self.declare_parameter('front_trigger_dist', FRONT_TRIGGER_DIST)
        self.front_trigger_dist = float(self.get_parameter('front_trigger_dist').value)

        self.declare_parameter('approach_speed', APPROACH_SPEED)
        self.approach_speed = float(self.get_parameter('approach_speed').value)

        self.declare_parameter('require_last_lane_trigger', True)
        self.require_last_lane_trigger = bool(self.get_parameter('require_last_lane_trigger').value)

        self.state = 'IDLE'
        self.records = []          # [(dt, linear_x, angular_z), ...] — dt = record_fixed_dt 로 고정
        self.pending_sample = None  # (vx, wz) — 아직 다음 메시지로 확정 안 된 "현재 유지 중"인 값

        self.replay_index = 0
        self.replay_elapsed = 0.0
        self.replay_last_t = None

        self.latest_scan = None
        self.auto_triggered = False   # 노드 하나당 한 번만 자동 발동
        self.last_lane_detected = not self.require_last_lane_trigger  # 게이트 안 쓰면 처음부터 통과

        if self.record_source == 'odom':
            self.create_subscription(Odometry, SUB_ODOM_IN, self.odom_callback, 10)
        else:
            self.create_subscription(Twist, SUB_CMD_VEL_IN, self.cmd_vel_callback, 10)
        self.create_subscription(Bool, SUB_RECORD_START, self.start_record_callback, 10)
        self.create_subscription(Bool, SUB_RECORD_STOP, self.stop_record_callback, 10)
        self.create_subscription(Bool, SUB_REPLAY_START, self.start_replay_callback, 10)
        self.create_subscription(Bool, SUB_REPLAY_CANCEL, self.cancel_callback, 10)
        self.create_subscription(LaserScan, SUB_SCAN, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Bool, SUB_LAST_LANE, self.last_lane_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.state_pub = self.create_publisher(String, PUB_STATE, 10)

        self.create_timer(1.0 / CONTROL_HZ, self.replay_loop)

        self.get_logger().info(
            f'cmd_vel Record/Replay Node Started (record_file={self.record_file}, '
            f'record_source={self.record_source}, auto_trigger={self.auto_trigger}, '
            f'front_trigger_dist={self.front_trigger_dist}m, approach_speed={self.approach_speed}m/s, '
            f'require_last_lane_trigger={self.require_last_lane_trigger})'
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ============================================================
    #  녹화
    # ============================================================
    def cmd_vel_callback(self, msg: Twist):
        self._record_sample(msg.linear.x, msg.angular.z)

    def odom_callback(self, msg: Odometry):
        self._record_sample(msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    # ============================================================
    #  자동 트리거 (라이다 전방 거리)
    # ============================================================
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def last_lane_callback(self, msg: Bool):
        if msg.data and not self.last_lane_detected:
            self.last_lane_detected = True
            self.get_logger().info('👀 last_lane_detected 수신 → 접근 주행 시작 가능')

    def _front_distance(self):
        scan = self.latest_scan
        if scan is None:
            return None
        dists = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                da = FRONT_ANGLE_CENTER - angle
                while da > math.pi:
                    da -= 2.0 * math.pi
                while da < -math.pi:
                    da += 2.0 * math.pi
                if abs(da) <= FRONT_ANGLE_HALF_WIDTH:
                    dists.append(r)
            angle += scan.angle_increment
        if not dists:
            return None
        dists.sort()
        return dists[len(dists) // 2]

    def _record_sample(self, vx, wz):
        # 새 명령이 도착한 시점 = 직전 명령을 "한 번의 키 입력"으로 확정하는 시점.
        # (teleop_twist_keyboard 는 키 누를 때만 한 번씩 발행하고 그 사이엔 무발행이므로,
        #  새 메시지 자신이 아니라 '직전' 값을 먼저 확정해야 재생 시 순서가 안 뒤집힌다.)
        # 실제 그 키를 얼마나 오래 눌렀는지는 재지 않고, 무조건 record_fixed_dt 로 통일한다.
        if self.state != 'RECORDING':
            return
        self._finalize_pending()
        self.pending_sample = (float(vx), float(wz))

    def _finalize_pending(self):
        if self.pending_sample is None:
            return
        vx, wz = self.pending_sample
        self.records.append((self.record_fixed_dt, vx, wz))
        self.pending_sample = None

    def start_record_callback(self, msg: Bool):
        if not msg.data or self.state != 'IDLE':
            return
        self.records = []
        self.pending_sample = None
        self.state = 'RECORDING'
        self.get_logger().info('⏺ RECORDING 시작')

    def stop_record_callback(self, msg: Bool):
        if not msg.data or self.state != 'RECORDING':
            return
        self._finalize_pending()
        self.state = 'IDLE'
        total_t = sum(dt for dt, _, _ in self.records)
        self._save_to_file()
        self.get_logger().info(
            f'⏹ RECORDING 종료: samples={len(self.records)}, total={total_t:.2f}s, '
            f'saved to {self.record_file}'
        )

    # ============================================================
    #  재생
    # ============================================================
    def start_replay_callback(self, msg: Bool):
        if not msg.data or self.state != 'IDLE':
            return
        self._trigger_replay('manual topic trigger')

    def _trigger_replay(self, reason: str):
        if not self.records:
            self._load_from_file()
        if not self.records:
            self.get_logger().warn('REPLAY 요청됐지만 재생할 기록이 없음 (녹화 먼저 하거나 record_file 확인)')
            return
        self.replay_index = 0
        self.replay_elapsed = 0.0
        self.replay_last_t = self.now_sec()
        self.state = 'REPLAYING'
        total_t = sum(dt for dt, _, _ in self.records)
        self.get_logger().info(
            f'▶ REPLAYING 시작 ({reason}): samples={len(self.records)}, total={total_t:.2f}s, '
            f'speed_scale={self.replay_speed_scale}x → 체감 {total_t / self.replay_speed_scale:.2f}s'
        )

    def cancel_callback(self, msg: Bool):
        if not msg.data:
            return
        if self.state in ('RECORDING', 'REPLAYING'):
            self.get_logger().warn(f'⚠️ {self.state} 강제 취소 → IDLE')
        self.state = 'IDLE'
        self.pending_sample = None
        self.cmd_pub.publish(Twist())

    def replay_loop(self):
        self._publish_state()

        if self.auto_trigger and self.state == 'IDLE' and not self.auto_triggered and not self.last_lane_detected:
            self.get_logger().warn(
                'IDLE: last_lane_detected 대기 중, 정지 유지', throttle_duration_sec=2.0
            )
            self.cmd_pub.publish(Twist())
            return

        if self.auto_trigger and self.state == 'IDLE' and not self.auto_triggered:
            front = self._front_distance()
            if front is not None and front <= self.front_trigger_dist:
                self.auto_triggered = True
                self._trigger_replay(f'auto lidar front={front:.2f}m')
            elif self.latest_scan is None:
                # /scan 자체를 아직 못 받음 -> 라이다 없이 맹목적으로 전진하지 않고 정지 대기
                self.get_logger().warn('IDLE: /scan 수신 대기 중, 정지 유지', throttle_duration_sec=2.0)
                self.cmd_pub.publish(Twist())
                return
            else:
                # 아직 트리거 거리(front_trigger_dist) 전 -> 그 지점까지 직진으로 접근
                approach = Twist()
                approach.linear.x = self.approach_speed
                self.get_logger().info(
                    f'IDLE→접근 중: front={"n/a" if front is None else f"{front:.2f}m"}, '
                    f'목표={self.front_trigger_dist}m',
                    throttle_duration_sec=1.0,
                )
                self.cmd_pub.publish(approach)
                return

        if self.state != 'REPLAYING':
            return

        t = self.now_sec()
        dt = 0.0 if self.replay_last_t is None else max(0.0, t - self.replay_last_t)
        self.replay_last_t = t
        self.replay_elapsed += dt * self.replay_speed_scale

        while self.replay_index < len(self.records):
            sample_dt = self.records[self.replay_index][0]
            if sample_dt <= 1e-3:
                self.replay_index += 1
                continue
            if self.replay_elapsed <= sample_dt:
                break
            self.replay_elapsed -= sample_dt
            self.replay_index += 1

        if self.replay_index >= len(self.records):
            self.state = 'IDLE'
            self.cmd_pub.publish(Twist())
            self.get_logger().info('✅ REPLAYING 완료 → IDLE')
            return

        _, vx, wz = self.records[self.replay_index]
        out = Twist()
        # 시간 압축(replay_elapsed)과 동일 배율로 속도도 올려야 궤적이 보존된 채 배속 재생됨.
        # (속도 배율 없이 시간만 압축하면 같은 명령을 짧게 유지 -> 이동량이 줄어 경로가 축소됨)
        out.linear.x = vx * self.replay_speed_scale
        out.angular.z = wz * self.replay_speed_scale
        self.cmd_pub.publish(out)

    # ============================================================
    #  파일 저장/로드
    #  - 재생에는 dt/vx/wz 세 값만 쓰지만, 사람이 눈으로 보기 쉽도록
    #    누적시간(t_cum)/방향(dir)/조향(steer) 주석 필드를 같이 저장한다.
    # ============================================================
    @staticmethod
    def _dir_word(vx):
        if vx > 0:
            return '전진'
        if vx < 0:
            return '후진'
        return '정지'

    @staticmethod
    def _steer_word(wz):
        if wz > 0:
            return '좌회전'
        if wz < 0:
            return '우회전'
        return '직진'

    def _save_to_file(self):
        try:
            os.makedirs(os.path.dirname(self.record_file), exist_ok=True)
            annotated = []
            for i, (dt, vx, wz) in enumerate(self.records):
                annotated.append({
                    'i': i,
                    'dt': round(dt, 4),
                    'vx': vx,
                    'wz': wz,
                    'dir': self._dir_word(vx),
                    'steer': self._steer_word(wz),
                })
            with open(self.record_file, 'w') as f:
                json.dump({'records': annotated}, f, indent=2, ensure_ascii=False)
        except OSError as e:
            self.get_logger().error(f'녹화 파일 저장 실패: {e}')

    def _load_from_file(self):
        try:
            with open(self.record_file, 'r') as f:
                data = json.load(f)
            records = []
            for r in data.get('records', []):
                if isinstance(r, dict):
                    records.append((float(r['dt']), float(r['vx']), float(r['wz'])))
                else:
                    dt, vx, wz = r
                    records.append((float(dt), float(vx), float(wz)))
            self.records = records
            self.get_logger().info(f'녹화 파일 로드: {self.record_file} ({len(self.records)} samples)')
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().warn(f'녹화 파일 로드 실패: {e}')
            self.records = []

    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRecordReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
