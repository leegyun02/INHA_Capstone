#!/usr/bin/env python3
"""
best.pt(PyTorch) -> TensorRT FP16(.engine) 변환 스크립트.

TensorRT 엔진은 GPU 아키텍처 + TensorRT/CUDA 버전에 종속적이라,
반드시 '실제로 추론을 돌릴 그 장비(Jetson)'에서 이 스크립트를 실행해야 합니다.
git으로 받은 .engine이 로드되지 않으면(역직렬화 에러) 십중팔구 이 종속성 불일치입니다.

사용법:
    pip install ultralytics        # 없으면 먼저 설치
    python3 weights/build_engine.py

결과:
    weights/best.engine  (ultralytics가 생성)
    ../obstacle.engine   (yolov8n_node가 참조하는 경로로 자동 복사)
"""

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# --------------- 설정 (yolov8n_node.py와 일치시킬 것) ---------------
HERE        = Path(__file__).resolve().parent          # .../trinity_ws/weights
PT_PATH     = HERE / "best.pt"
IMGSZ       = 640     # yolov8n_node.py IMGSZ
DEVICE      = 0       # Jetson GPU
HALF        = True    # FP16(16비트) 양자화
WORKSPACE   = 4       # 빌드 시 임시 워크스페이스(GB). 메모리 부족하면 낮추세요.
# 노드가 참조하는 최종 경로 (yolov8n_node.py MODEL_PATH)
DEST_ENGINE = HERE.parent / "obstacle.engine"
# -------------------------------------------------------------------


def main() -> int:
    if not PT_PATH.exists():
        print(f"[ERROR] 가중치 파일이 없습니다: {PT_PATH}")
        return 1

    print(f"[1/3] 로드: {PT_PATH}")
    model = YOLO(str(PT_PATH))
    print(f"      클래스: {model.names}")

    print(f"[2/3] TensorRT 엔진 빌드 (imgsz={IMGSZ}, half={HALF}, device={DEVICE}) ...")
    print("      Jetson에서는 수 분 소요될 수 있습니다.")
    exported = model.export(
        format="engine",
        imgsz=IMGSZ,
        half=HALF,          # FP16
        device=DEVICE,
        workspace=WORKSPACE,
        verbose=True,
    )
    exported = Path(exported)
    print(f"      생성됨: {exported}")

    print(f"[3/3] 노드 참조 경로로 복사: {DEST_ENGINE}")
    shutil.copyfile(exported, DEST_ENGINE)

    print("\n[완료]")
    print(f"  엔진: {exported}")
    print(f"  배포: {DEST_ENGINE}")
    print("  이제 `ros2 run obstacle_detection_package yolov8n_node` 로 확인하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
