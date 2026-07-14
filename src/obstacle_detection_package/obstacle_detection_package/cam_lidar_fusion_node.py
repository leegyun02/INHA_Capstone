import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                     튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_YOLO_TOPIC   = '/detections'
SUB_SCAN_TOPIC   = '/scan'
PUB_FUSED_TOPIC  = '/obstacles/fused'
PUB_MARKER_TOPIC = '/obstacles/markers'

# --- 카메라 ---
IMG_W            = 640.0
CAMERA_FOV_DEG   = 66.74

# --- 클래스 ---
TARGET_CLASSES         = ['Cone', 'Person', 'Box', 'Car']  # 추적 대상 (Tunnel 제외)
TUNNEL_CLASS           = 'Tunnel'
MULTI_INSTANCE_CLASSES = ['Cone']   # 다중 허용 클래스. 나머지는 장면당 1개만 유지

# --- 유효 영역 필터 ---
MAX_RANGE        = 2.0      # 이 거리 밖 장애물 무시 [m]
FORWARD_MIN      = 0.1      # 이 값보다 뒤(작으면) 무시 [m] (후방/측면 제거)

# --- 씨앗 트래킹 (Person, Car 전용) ---
SEED_RADIUS      = 0.35     # 씨앗 근처 탐색 반경 [m]
LOST_FRAMES      = 10       # Lost 판정 프레임 (20Hz * 10 = 0.5초)
MIN_POINTS       = 4        # 클러스터 최소 점 개수
MATCH_DIST       = 0.4      # YOLO 검출 ↔ 기존 트래커 매칭 거리 [m]
SCAN_WINDOW      = 5        # YOLO 각도 주변 스캔 탐색 폭

# --- 터널 벽 인식 (Seed 확정 + 코리더 폭 게이트) ---
#   [Seed]   YOLO bbox 각도 창 안 라이다가 있으면 '터널 실재' 확정 (전역 최소 1개 요구).
#            → YOLO 오검출 + 무관 물체만 있는 프레임을 통째로 배제.
#   [Expand] 넓은 전방 콘 안 인접빔 연결 클러스터링 → 벽은 bbox 밖 입구까지 한 덩어리.
#   [Gate]   좌/우 벽은 (코리더 전방 + 터널 폭 이내 |y|<=MAX_OFFSET) 조건으로 선택.
#            seed 포함 클러스터를 우선하되, 반대쪽 벽은 seed 없이도 폭 게이트로 복구.
#            → bbox가 작아 한쪽 벽만 프레임에 담겨도 반대 벽을 놓치지 않음.
#            → 터널 폭 밖 옆 장애물은 폭 게이트에서 걸러짐.
TUNNEL_MAX_RANGE       = 2.0    # 터널 벽 최대 거리 [m]
TUNNEL_ANGLE_PAD_DEG   = 3.0    # bbox 씨앗 각도 창 좌우 여유 [deg]
TUNNEL_SEARCH_HALF_DEG = 80.0   # 확장 탐색 전방 콘 반각 [deg] (hint 중심)
TUNNEL_CLUSTER_BREAK   = 0.15   # 클러스터 경계 거리 점프 [m]
TUNNEL_BRIDGE_GAP      = 3      # 무효빔 브리징 허용 (인덱스 갭)
TUNNEL_WALL_MIN_POINTS = 3      # 벽으로 인정할 클러스터 최소 점 개수
TUNNEL_WALL_NEAR_K     = 3      # 벽 대표점 계산 시 사용할 입구쪽(최근접) 포인트 개수
TUNNEL_WALL_MAX_OFFSET = 0.7    # 코리더 축 기준 벽 최대 횡오프셋 [m] (터널 반폭+여유)
                                #   이보다 옆으로 벗어난 클러스터는 벽 후보에서 제외

# --- RViz 마커 ---
PUBLISH_DEBUG    = True     # 마커 발행 on/off
MARKER_LIFETIME  = 0.2       # 마커 유지 시간 [s]
MARKER_DIAMETER  = 0.3       # 실린더 지름 [m]

# --- 기타 ---
YOLO_TIMEOUT     = 0.3       # YOLO 검출 유효 시간 [s]
LASER_FRAME_ID   = 'laser_link'
RATE_HZ          = 20.0      # 발행 주기 [Hz]
# ============================================================


class Tracker:
    """장애물 하나의 추적 상태"""
    _next_id = 0

    def __init__(self, class_name, pos):
        self.id = Tracker._next_id
        Tracker._next_id += 1
        self.class_name = class_name
        self.pos = pos          # (fx, fy) 미터
        self.lost_count = 0
        self.source = 'fusion'


class CameraLidarFusionNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_fusion_node')

        self.camera_fov_rad = CAMERA_FOV_DEG * (math.pi / 180.0)
        self.tunnel_pad_rad = TUNNEL_ANGLE_PAD_DEG * (math.pi / 180.0)
        self.tunnel_search_half_rad = TUNNEL_SEARCH_HALF_DEG * (math.pi / 180.0)

        self.latest_scan = None
        self.latest_detections = []      # [(class_name, cx_pixel, bbox_w), ...]
        self.latest_det_time = self.get_clock().now()

        self.trackers = []

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.scan_sub = self.create_subscription(LaserScan, SUB_SCAN_TOPIC, self.scan_callback, qos)
        self.yolo_sub = self.create_subscription(Detection2DArray, SUB_YOLO_TOPIC, self.yolo_callback, qos)

        self.fused_pub = self.create_publisher(String, PUB_FUSED_TOPIC, 10)
        if PUBLISH_DEBUG:
            self.marker_pub = self.create_publisher(MarkerArray, PUB_MARKER_TOPIC, 10)

        self.create_timer(1.0 / RATE_HZ, self.fuse_and_publish)
        self.get_logger().info('Camera-LiDAR Fusion 시작')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def yolo_callback(self, msg: Detection2DArray):
        dets = []
        for det in msg.detections:
            if not det.results:
                continue
            cls = det.results[0].hypothesis.class_id     # 문자열 클래스명
            cx = det.bbox.center.position.x               # 박스 중심 x (픽셀)
            bw = det.bbox.size_x                          # 박스 폭 (픽셀)
            dets.append((cls, cx, bw))
        self.latest_detections = dets
        self.latest_det_time = self.get_clock().now()

    # ================= 메인 루프 =================
    def fuse_and_publish(self):
        scan = self.latest_scan
        if scan is None:
            return

        age = (self.get_clock().now() - self.latest_det_time).nanoseconds * 1e-9
        detections = self.latest_detections if age < YOLO_TIMEOUT else []

        # 클래스 우선순위 억제: Car가 보이면 Cone 무시, Cone이 보이면 Box 무시
        if any(cls == 'Car' for cls, _, _ in detections):
            detections = [d for d in detections if d[0] != 'Cone']
        if any(cls == 'Cone' for cls, _, _ in detections):
            detections = [d for d in detections if d[0] != 'Box']

        # 터널 감지 + YOLO Car 인식 여부 (behavior 신호용)
        tunnel_det = self._find_tunnel(detections)  # (cx, bw) or None
        tunnel_on = tunnel_det is not None
        yolo_car_seen = any(cls == 'Car' for cls, _, _ in detections)

        # 1) YOLO 검출 → 차량 기준 좌표(m). 터널 중이면 Cone/Box 제외
        yolo_obs = self._yolo_to_positions(detections, scan, tunnel_on)
        yolo_obs = self._keep_closest_single(yolo_obs)   # Cone 외엔 가장 가까운 1개만

        # 2) YOLO 검출로 트래커 갱신/생성 (우선권)
        matched = set()
        for cls, pos in yolo_obs:
            tr = self._match_tracker(cls, pos)
            if tr is None:
                tr = Tracker(cls, pos)
                self.trackers.append(tr)
            else:
                tr.pos = pos
            tr.lost_count = 0
            tr.source = 'fusion'
            matched.add(tr.id)

        # 3) YOLO가 못 본 트래커 → 씨앗 근처 클러스터로 추적 유지 (Person, Car 한정)
        survivors = []
        for tr in self.trackers:
            if tr.id in matched:
                survivors.append(tr)
                continue

            # Box는 YOLO에서 놓치면 라이다 추적 없이 즉시 폐기
            if tr.class_name == 'Box':
                continue

            new_pos = self._find_cluster_near(scan, tr.pos, SEED_RADIUS)
            if new_pos is not None:
                if new_pos[0] <= FORWARD_MIN:
                    continue
                tr.pos = new_pos
                tr.lost_count = 0
                tr.source = 'lidar'
                survivors.append(tr)
            else:
                tr.lost_count += 1
                if tr.lost_count < LOST_FRAMES:
                    tr.source = 'lidar'
                    survivors.append(tr)
        self.trackers = survivors

        # 3.5) 단일 인스턴스 강제: Cone 외 클래스는 1개만, 중복 시 fusion 우선
        self._enforce_single_instance()

        # 4) 발행 (JSON) — 전방 + 2m 이내만
        combined = [{
            'class': tr.class_name,
            'source': tr.source,
            'distance_r': round(math.hypot(tr.pos[0], tr.pos[1]), 3),
            'forward_x': round(tr.pos[0], 3),
            'lateral_y': round(tr.pos[1], 3),
        } for tr in self.trackers
          if tr.pos[0] > FORWARD_MIN and math.hypot(tr.pos[0], tr.pos[1]) <= MAX_RANGE]

        # 5) 터널 벽: bbox 씨앗으로 확정 → 코리더 폭 게이트로 좌/우 벽 (입구 최근접)
        #    터널 중엔 벽/TunnelActive만 발행하고 다른 인식(Cone/Person/Car/Box)은 전부 억제.
        #    (survivor 씨앗 트래킹으로 살아남은 Cone 등이 터널 중 새어나오는 걸 발행단에서 차단)
        if tunnel_on:
            walls = self._detect_tunnel_walls(tunnel_det, scan)
            combined = walls + [{'class': 'TunnelActive', 'source': 'yolo'}]
        else:
            if yolo_car_seen:
                combined.append({'class': 'CarDetected', 'source': 'yolo'})

        if not combined:
            return

        out = String()
        out.data = json.dumps(combined)
        self.fused_pub.publish(out)

        if PUBLISH_DEBUG:
            self.publish_rviz_markers(combined)

    # ================= 단일 인스턴스 강제 =================
    def _keep_closest_single(self, yolo_obs):
        """Cone 외 클래스는 프레임당 가장 가까운 fusion 검출 1개만 사용.
        서로 붙은 박스 2개가 같은 트래커에 덮어써지는 문제를 매칭 전에 차단."""
        best = {}   # cls -> (pos, dist)
        result = []
        for cls, pos in yolo_obs:
            if cls in MULTI_INSTANCE_CLASSES:
                result.append((cls, pos))
                continue
            d = math.hypot(pos[0], pos[1])
            if cls not in best or d < best[cls][1]:
                best[cls] = (pos, d)
        result.extend((cls, v[0]) for cls, v in best.items())
        return result

    def _enforce_single_instance(self):
        """Cone을 제외한 클래스는 장면당 1개만 유지.
        중복 시 fusion 우선 → lost_count 작은 것 → 가까운 것 순으로 선택."""
        by_class = {}
        keep = []
        for tr in self.trackers:
            if tr.class_name in MULTI_INSTANCE_CLASSES:
                keep.append(tr)                 # 다중 허용 클래스는 그대로
            else:
                by_class.setdefault(tr.class_name, []).append(tr)

        for group in by_class.values():
            group.sort(key=lambda t: (
                0 if t.source == 'fusion' else 1,   # fusion 우선
                t.lost_count,                       # 최근 갱신 우선
                math.hypot(t.pos[0], t.pos[1]),     # 가까운 것 우선
            ))
            keep.append(group[0])                   # 하나만 남기고 나머지 폐기

        self.trackers = keep

    # ================= 터널 벽 인식 =================
    def _find_tunnel(self, detections):
        best = None
        for cls, cx, bw in detections:
            if cls == TUNNEL_CLASS and (best is None or bw > best[1]):
                best = (cx, bw)
        return best

    @staticmethod
    def _norm_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _cluster_forward_points(self, pts):
        """인덱스 순서 pts=[(i, r, angle, in_box), ...] 를
        거리 점프/무효빔 갭 기준으로 분할. (in_box 등 뒤 필드는 그대로 보존)"""
        clusters = []
        cur = [pts[0]]
        for k in range(1, len(pts)):
            prev_r = cur[-1][1]
            r = pts[k][1]
            idx_gap = pts[k][0] - cur[-1][0]
            if abs(r - prev_r) > TUNNEL_CLUSTER_BREAK or idx_gap > (TUNNEL_BRIDGE_GAP + 1):
                clusters.append(cur)
                cur = [pts[k]]
            else:
                cur.append(pts[k])
        clusters.append(cur)
        return clusters

    @staticmethod
    def _tunnel_wall_entry(fx, fy, side):
        return {
            'class': 'TunnelWall',
            'side': side,
            'source': 'lidar',
            'distance_r': round(math.hypot(fx, fy), 3),
            'forward_x': round(fx, 3),
            'lateral_y': round(fy, 3),
        }

    def _detect_tunnel_walls(self, tunnel_det, scan):
        cx, bw = tunnel_det

        # --- bbox 씨앗 각도 창 (터널 실재 확인용) ---
        x_left = cx - bw / 2.0
        x_right = cx + bw / 2.0
        ang_a = -(x_left - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ang_b = -(x_right - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        box_lo = min(ang_a, ang_b) - self.tunnel_pad_rad
        box_hi = max(ang_a, ang_b) + self.tunnel_pad_rad

        # --- 코리더 축(hint): 좌/우 분류 + 전방/횡오프셋 계산 기준 ---
        hint = -(cx - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ch, sh = math.cos(hint), math.sin(hint)

        amin = scan.angle_min
        ainc = scan.angle_increment
        rmin = scan.range_min

        # 1) 확장 탐색 콘(넓게) 안 유효 라이다. 각 점이 bbox 씨앗인지(in_box) 표시.
        pts = []
        for i, r in enumerate(scan.ranges):
            if not (math.isfinite(r) and rmin < r < TUNNEL_MAX_RANGE):
                continue
            angle = amin + i * ainc
            if abs(self._norm_angle(angle - hint)) > self.tunnel_search_half_rad:
                continue
            in_box = (box_lo <= angle <= box_hi)
            pts.append((i, r, angle, in_box))

        if len(pts) < TUNNEL_WALL_MIN_POINTS:
            return []

        # 2) 인접빔 연결 클러스터링. 벽은 bbox 밖 입구까지 한 덩어리로 이어짐.
        clusters = self._cluster_forward_points(pts)

        # 3) 클러스터 요약 + 게이트(코리더 전방 & 터널 폭 이내) → 좌/우 후보
        #    대표점 = 입구쪽 최근접 K점 평균. seed = bbox 창 포인트 포함 여부.
        left_cands, right_cands = [], []
        for c in clusters:
            if len(c) < TUNNEL_WALL_MIN_POINTS:
                continue
            near = sorted(c, key=lambda p: p[1])[:TUNNEL_WALL_NEAR_K]
            fx = sum(p[1] * math.cos(p[2]) for p in near) / len(near)
            fy = sum(p[1] * math.sin(p[2]) for p in near) / len(near)
            x_corr = fx * ch + fy * sh            # 코리더 전방 (양수여야 앞쪽)
            y_corr = -fx * sh + fy * ch           # 코리더 축 기준 횡오프셋
            if x_corr <= 0.0:
                continue                          # 코리더 뒤쪽 클러스터 제외
            if abs(y_corr) > TUNNEL_WALL_MAX_OFFSET:
                continue                          # 터널 폭 밖 = 옆 장애물, 제외
            cand = {'fx': fx, 'fy': fy, 'n': len(c), 'seed': any(p[3] for p in c)}
            (left_cands if y_corr >= 0.0 else right_cands).append(cand)

        # 4) 전역 seed 최소 1개: 라이다로 터널 벽이 실제 확인되어야 발행 (오검출 방지)
        if not any(cd['seed'] for cd in left_cands + right_cands):
            return []

        # 5) 각 side 선택: seed 포함 우선 → 점 많은 순.
        #    (반대쪽 벽은 seed 없어도 폭 게이트 통과분 중 최선으로 복구)
        def pick(cands):
            if not cands:
                return None
            return max(cands, key=lambda s: (1 if s['seed'] else 0, s['n']))

        walls = []
        l = pick(left_cands)
        r = pick(right_cands)
        if l is not None:
            walls.append(self._tunnel_wall_entry(l['fx'], l['fy'], 'left'))
        if r is not None:
            walls.append(self._tunnel_wall_entry(r['fx'], r['fy'], 'right'))
        return walls

    # ================= YOLO → 위치 =================
    def _yolo_to_positions(self, detections, scan, tunnel_on=False):
        result = []
        for cls, cx, bw in detections:
            if cls not in TARGET_CLASSES:
                continue
            if tunnel_on and cls in ('Cone', 'Box'):   # 터널 중엔 Cone/Box 무시
                continue

            if cls in ('Cone', 'Box'):
                # bbox 영역 안 라이다 점을 하나의 좌표로 묶음
                pos = self._get_cluster_from_bbox(cx, bw, scan)
                if pos is not None and math.hypot(*pos) <= MAX_RANGE:
                    result.append((cls, pos))
            else:
                # Person, Car: bbox 중심 각도 주변 최근접 거리
                angle = -(cx - (IMG_W / 2.0)) * (self.camera_fov_rad / IMG_W)
                distance = self.get_distance_from_scan(angle, scan)
                if distance is not None and distance <= MAX_RANGE:
                    result.append((cls, (distance * math.cos(angle),
                                         distance * math.sin(angle))))
        return result

    def _get_cluster_from_bbox(self, cx, bw, scan):
        x_left = cx - bw / 2.0
        x_right = cx + bw / 2.0
        ang_a = -(x_left - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ang_b = -(x_right - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ang_lo = min(ang_a, ang_b)
        ang_hi = max(ang_a, ang_b)

        amin = scan.angle_min
        ainc = scan.angle_increment
        rmin = scan.range_min
        rmax = min(scan.range_max, MAX_RANGE)

        xs, ys = [], []
        for i, r in enumerate(scan.ranges):
            if not (math.isfinite(r) and rmin < r < rmax):
                continue
            angle = amin + i * ainc
            if ang_lo <= angle <= ang_hi:
                xs.append(r * math.cos(angle))
                ys.append(r * math.sin(angle))

        if not xs:
            return None
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _match_tracker(self, cls, pos):
        best, best_d = None, MATCH_DIST
        for tr in self.trackers:
            if tr.class_name != cls:
                continue
            d = math.hypot(tr.pos[0] - pos[0], tr.pos[1] - pos[1])
            if d < best_d:
                best, best_d = tr, d
        return best

    def _find_cluster_near(self, scan, seed, radius):
        sx, sy = seed
        amin = scan.angle_min
        ainc = scan.angle_increment
        rmin = scan.range_min
        rmax = scan.range_max

        xs, ys = [], []
        for i, r in enumerate(scan.ranges):
            if not (math.isfinite(r) and rmin < r < rmax):
                continue
            angle = amin + i * ainc
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            if math.hypot(px - sx, py - sy) <= radius:
                xs.append(px)
                ys.append(py)

        if len(xs) < MIN_POINTS:
            return None
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def get_distance_from_scan(self, target_angle, scan):
        if target_angle < scan.angle_min or target_angle > scan.angle_max:
            return None
        idx = int((target_angle - scan.angle_min) / scan.angle_increment)
        start_idx = max(0, idx - SCAN_WINDOW)
        end_idx = min(len(scan.ranges), idx + SCAN_WINDOW + 1)
        valid = [scan.ranges[i] for i in range(start_idx, end_idx)
                 if scan.range_min < scan.ranges[i] < scan.range_max]
        return min(valid) if valid else None

    # ================= RViz 마커 =================
    def publish_rviz_markers(self, obstacles):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        life_sec = int(MARKER_LIFETIME)
        life_nsec = int((MARKER_LIFETIME - life_sec) * 1e9)

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        idx = 0
        for obs in obstacles:
            if 'forward_x' not in obs:
                continue
            is_wall = obs.get('class') == 'TunnelWall'
            is_lidar = obs.get('source') == 'lidar'

            cyl = Marker()
            cyl.header.frame_id = LASER_FRAME_ID
            cyl.header.stamp = current_time
            cyl.ns = 'obstacles_shape'
            cyl.id = idx * 2
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = float(obs['forward_x'])
            cyl.pose.position.y = float(obs['lateral_y'])
            cyl.pose.position.z = 0.5
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = MARKER_DIAMETER
            cyl.scale.y = MARKER_DIAMETER
            cyl.scale.z = 1.0
            if is_wall:
                cyl.color.r, cyl.color.g, cyl.color.b = 0.0, 0.4, 1.0
            else:
                cyl.color.r = 1.0
                cyl.color.g = 0.5 if is_lidar else 0.0
                cyl.color.b = 0.0
            cyl.color.a = 0.7
            cyl.lifetime.sec = life_sec
            cyl.lifetime.nanosec = life_nsec

            text = Marker()
            text.header.frame_id = LASER_FRAME_ID
            text.header.stamp = current_time
            text.ns = 'obstacles_text'
            text.id = idx * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(obs['forward_x'])
            text.pose.position.y = float(obs['lateral_y'])
            text.pose.position.z = 1.2
            text.pose.orientation.w = 1.0
            label = obs['class'] + (f"({obs.get('side','')})" if is_wall else f"({obs.get('source','')})")
            text.text = f"{label}\n{obs['distance_r']}m"
            text.scale.z = 0.3
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 0.0, 1.0
            text.lifetime.sec = life_sec
            text.lifetime.nanosec = life_nsec

            marker_array.markers.append(cyl)
            marker_array.markers.append(text)
            idx += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Camera-LiDAR Fusion 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()