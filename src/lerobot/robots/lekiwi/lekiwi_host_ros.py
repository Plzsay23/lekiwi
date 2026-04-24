#!/usr/bin/env python

import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


@dataclass
class LeKiwiServerConfig:
    """Configuration for the LeKiwi host ROS script."""
    robot: LeKiwiConfig = field(default_factory=LeKiwiConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz
        
        logging.info(
            "LeKiwiHost sockets bound: cmd=tcp://*:%s obs=tcp://*:%s",
            config.port_zmq_cmd,
            config.port_zmq_observations,
        )
        logging.info(
            "LeKiwiHost config: connection_time_s=%s watchdog_timeout_ms=%s max_loop_freq_hz=%s",
            self.connection_time_s,
            self.watchdog_timeout_ms,
            self.max_loop_freq_hz,
        )

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _color_mode_string(cam_cfg) -> str:
    mode = getattr(cam_cfg, "color_mode", None)
    if mode is None:
        return "rgb"
    return str(getattr(mode, "value", mode)).lower()


def _is_bgr_mode(cam_cfg) -> bool:
    return _color_mode_string(cam_cfg) == "bgr"


def _depth_to_preview_bgr(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)

    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]

    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape={depth.shape}")

    if np.issubdtype(depth.dtype, np.floating):
        finite_mask = np.isfinite(depth)
        if not finite_mask.any():
            scaled = np.zeros(depth.shape, dtype=np.uint8)
        else:
            valid = depth[finite_mask]
            dmin = float(valid.min())
            dmax = float(valid.max())
            if dmax <= dmin:
                scaled = np.zeros(depth.shape, dtype=np.uint8)
            else:
                scaled = np.zeros(depth.shape, dtype=np.float32)
                scaled[finite_mask] = (depth[finite_mask] - dmin) * 255.0 / (dmax - dmin)
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        depth_f = depth.astype(np.float32)
        dmin = float(depth_f.min()) if depth_f.size else 0.0
        dmax = float(depth_f.max()) if depth_f.size else 0.0
        if dmax <= dmin:
            scaled = np.zeros(depth.shape, dtype=np.uint8)
        else:
            scaled = np.clip((depth_f - dmin) * 255.0 / (dmax - dmin), 0, 255).astype(np.uint8)

    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


class LeKiwiRosPublisher(Node):
    JOINT_STATE_KEYS = [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
    ]

    JOINT_NAMES = [
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    ]

    def __init__(self, camera_configs: dict):
        super().__init__("lekiwi_host_ros")
        self.bridge = CvBridge()
        self.camera_configs = camera_configs

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.base_vel_pub = self.create_publisher(TwistStamped, "/lekiwi/base/velocity", 10)

        self.rgb_pubs = {
            cam_name: self.create_publisher(Image, f"/lekiwi/camera/{cam_name}/rgb/image_raw", 10)
            for cam_name in camera_configs.keys()
        }

        self.depth_raw_pubs = {
            cam_name: self.create_publisher(Image, f"/lekiwi/camera/{cam_name}/depth/image_raw", 10)
            for cam_name, cfg in camera_configs.items()
            if getattr(cfg, "use_depth", False)
        }

        self.depth_preview_pubs = {
            cam_name: self.create_publisher(Image, f"/lekiwi/camera/{cam_name}/depth/image_preview", 10)
            for cam_name, cfg in camera_configs.items()
            if getattr(cfg, "use_depth", False)
        }

    def publish_joint_and_base(self, observation: dict):
        stamp = self.get_clock().now().to_msg()

        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.header.frame_id = "lekiwi_base"
        joint_msg.name = self.JOINT_NAMES
        joint_msg.position = [float(observation.get(k, 0.0)) for k in self.JOINT_STATE_KEYS]
        joint_msg.velocity = []
        joint_msg.effort = []
        self.joint_pub.publish(joint_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = "lekiwi_base"
        vel_msg.twist.linear.x = float(observation.get("x.vel", 0.0))
        vel_msg.twist.linear.y = float(observation.get("y.vel", 0.0))
        vel_msg.twist.linear.z = 0.0

        theta_deg = float(observation.get("theta.vel", 0.0))
        vel_msg.twist.angular.x = 0.0
        vel_msg.twist.angular.y = 0.0
        vel_msg.twist.angular.z = np.deg2rad(theta_deg)
        self.base_vel_pub.publish(vel_msg)

    def publish_rgb(self, cam_name: str, image: np.ndarray):
        cam_cfg = self.camera_configs[cam_name]
        encoding = "bgr8" if _is_bgr_mode(cam_cfg) else "rgb8"

        msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(image), encoding=encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{cam_name}_rgb_optical_frame"
        self.rgb_pubs[cam_name].publish(msg)

    def publish_depth(self, cam_name: str, depth: np.ndarray):
        if cam_name not in self.depth_raw_pubs:
            return

        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[..., 0]

        if depth.dtype == np.uint16:
            encoding = "16UC1"
        elif depth.dtype == np.float32:
            encoding = "32FC1"
        elif depth.dtype == np.float64:
            depth = depth.astype(np.float32)
            encoding = "32FC1"
        elif depth.dtype == np.uint8:
            encoding = "mono8"
        else:
            depth = depth.astype(np.float32)
            encoding = "32FC1"

        raw_msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(depth), encoding=encoding)
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = f"{cam_name}_depth_optical_frame"
        self.depth_raw_pubs[cam_name].publish(raw_msg)

        preview = _depth_to_preview_bgr(depth)
        preview_msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(preview), encoding="bgr8")
        preview_msg.header.stamp = self.get_clock().now().to_msg()
        preview_msg.header.frame_id = f"{cam_name}_depth_optical_frame"
        self.depth_preview_pubs[cam_name].publish(preview_msg)


def _get_depth_from_camera_obj(cam) -> np.ndarray | None:
    if not hasattr(cam, "latest_depth_frame") or not hasattr(cam, "frame_lock"):
        return None

    with cam.frame_lock:
        depth = cam.latest_depth_frame
        if depth is None:
            return None
        return depth.copy()


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring LeKiwi")
    robot = LeKiwi(cfg.robot)

    logging.info("Connecting LeKiwi")
    robot.connect()
    
    logging.info("LeKiwi connected successfully.")
    logging.info("Connected cameras: %s", list(robot.cameras.keys()))
    for cam_name, cam_cfg in cfg.robot.cameras.items():
        logging.info("Camera '%s' config: %s", cam_name, cam_cfg)

    logging.info("Starting HostAgent")
    host = LeKiwiHost(cfg.host)

    rclpy.init(args=None)
    ros_pub = LeKiwiRosPublisher(camera_configs=cfg.robot.cameras)

    last_cmd_time = time.time()
    logging.info("LeKiwi host server is up.")
    logging.info("Waiting for commands on tcp://*:%s", cfg.host.port_zmq_cmd)
    logging.info("Publishing observations on tcp://*:%s", cfg.host.port_zmq_observations)

    start = time.perf_counter()
    
    loop_started_logged = False

    try:
        while True:
            if host.connection_time_s > 0:
                duration = time.perf_counter() - start
                if duration >= host.connection_time_s:
                    logging.info("Connection time reached. Exiting host loop.")
                    break

            loop_start_time = time.time()
            
            if not loop_started_logged:
                logging.info("Main host loop started.")
                loop_started_logged = True

            try:
                try:
                    msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                    data = dict(json.loads(msg))
                    _action_sent = robot.send_action(data)
                    last_cmd_time = time.time()
                except zmq.Again:
                    pass
                except Exception as e:
                    logging.error("Message fetching failed: %s", e)

                if host.watchdog_timeout_ms > 0:
                    now = time.time()
                    if now - last_cmd_time > host.watchdog_timeout_ms / 1000:
                        logging.warning(
                            "Command not received for more than %s milliseconds. Stopping the base.",
                            host.watchdog_timeout_ms,
                        )
                        robot.stop_base()
                        last_cmd_time = now

                observation = robot.get_observation()

                ros_pub.publish_joint_and_base(observation)

                for cam_name, cam_cfg in cfg.robot.cameras.items():
                    rgb = observation.get(cam_name)
                    if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] == 3:
                        ros_pub.publish_rgb(cam_name, rgb)

                    if getattr(cam_cfg, "use_depth", False):
                        cam_obj = robot.cameras[cam_name]
                        depth = _get_depth_from_camera_obj(cam_obj)
                        if isinstance(depth, np.ndarray):
                            ros_pub.publish_depth(cam_name, depth)

                rclpy.spin_once(ros_pub, timeout_sec=0.0)

            except Exception as e:
                logging.exception("Main host ROS loop error: %s", e)
                time.sleep(0.1)

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        try:
            robot.disconnect()
        finally:
            host.disconnect()
            ros_pub.destroy_node()
            rclpy.shutdown()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    main()