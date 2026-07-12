#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trinity_bringup.launch.py

Trinity 자율주행 통합 브링업 런치.

실행 순서:
  1) obstacle_detection_package / yolov8n_node          (YOLO 객체 탐지)
  2) obstacle_detection_package / cam_lidar_fusion_node  (카메라-라이다 융합)
  3) obstacle_detection_package / traffic_light_perception_node (신호등 인지)
  4) decision_making_package    / stanley_follow         (차선 추종 조향)
  5) decision_making_package    / final_test_state_machine (행동 계획 / 상태머신)

주요 런치 인자 (ros2 launch bringup_trinity trinity_bringup.launch.py <arg>:=<value>):
  - model_path            : YOLO 모델(pt/engine) 파일 경로
  - debug                 : 상태머신 디버그 로그 (true/false)
  - enable_traffic_light  : 신호등(초록불 출발) 미션 (true/false)
  - enable_car_follow     : 앞차 추종 미션 (true/false)
  - enable_cone           : 콘 갈림길 주행 미션 (true/false)
  - enable_tunnel         : 터널 주행 미션 (true/false)
  - enable_person         : 사람 정지 미션 (true/false)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ------------------------------------------------------------------
    #  런치 인자 선언
    # ------------------------------------------------------------------
    # model_path 는 문자열, 나머지 미션/디버그 플래그는 bool 타입으로 강제 변환
    # ("true"/"false" 문자열이 bool 파라미터에 그대로 들어가는 문제 방지)
    model_path = LaunchConfiguration('model_path')
    debug = ParameterValue(LaunchConfiguration('debug'), value_type=bool)
    enable_traffic_light = ParameterValue(
        LaunchConfiguration('enable_traffic_light'), value_type=bool)
    enable_car_follow = ParameterValue(
        LaunchConfiguration('enable_car_follow'), value_type=bool)
    enable_cone = ParameterValue(LaunchConfiguration('enable_cone'), value_type=bool)
    enable_tunnel = ParameterValue(LaunchConfiguration('enable_tunnel'), value_type=bool)
    enable_person = ParameterValue(LaunchConfiguration('enable_person'), value_type=bool)

    declare_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/wego/trinity_ws/obstacle.engine',
            description='YOLO 모델(pt/engine) 파일 경로',
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='상태머신(final_test_state_machine) 디버그 로그 On/Off',
        ),
        DeclareLaunchArgument(
            'enable_traffic_light',
            default_value='false',
            description='신호등(초록불 출발) 미션 On/Off',
        ),
        DeclareLaunchArgument(
            'enable_car_follow',
            default_value='true',
            description='앞차 추종(속도 캡) 미션 On/Off',
        ),
        DeclareLaunchArgument(
            'enable_cone',
            default_value='true',
            description='콘 갈림길 주행 미션 On/Off',
        ),
        DeclareLaunchArgument(
            'enable_tunnel',
            default_value='true',
            description='터널 주행 미션 On/Off',
        ),
        DeclareLaunchArgument(
            'enable_person',
            default_value='true',
            description='사람 정지 미션 On/Off',
        ),
    ]

    # ------------------------------------------------------------------
    #  노드 (실행 순서대로 나열)
    # ------------------------------------------------------------------
    # 1) YOLO 객체 탐지 (모델 경로를 파라미터로 조정 가능)
    yolo_node = Node(
        package='obstacle_detection_package',
        executable='yolov8n_node',
        name='yolo_detect_node',
        output='screen',
        parameters=[{'model_path': model_path}],
    )

    # 2) 카메라-라이다 융합
    fusion_node = Node(
        package='obstacle_detection_package',
        executable='cam_lidar_fusion_node',
        name='camera_lidar_fusion_node',
        output='screen',
    )

    # 3) 신호등 인지
    # traffic_light_node = Node(
    #     package='obstacle_detection_package',
    #     executable='traffic_light_perception_node',
    #     name='traffic_light_perception_node',
    #     output='screen',
    # )

    # 4) 차선 추종 (Stanley)
    stanley_node = Node(
        package='decision_making_package',
        executable='my_test_stanley',
        name='lane_follow',
        output='screen',
    )

    # 5) 행동 계획 / 상태머신 (디버그 및 미션별 On/Off 파라미터)
    state_machine_node = Node(
        package='decision_making_package',
        executable='final_test_state_machine',
        name='behavior_planner_node',
        output='screen',
        parameters=[{
            'debug': debug,
            'enable_traffic_light': enable_traffic_light,
            'enable_car_follow': enable_car_follow,
            'enable_cone': enable_cone,
            'enable_tunnel': enable_tunnel,
            'enable_person': enable_person,
        }],
    )

    return LaunchDescription(declare_args + [
        yolo_node,
        fusion_node,
        # traffic_light_node,
        stanley_node,
        state_machine_node,
    ])
