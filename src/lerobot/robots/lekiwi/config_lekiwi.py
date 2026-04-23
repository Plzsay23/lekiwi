from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config_opencv() -> dict[str, CameraConfig]:
    return {
        "top": OpenCVCameraConfig(
            index_or_path="/dev/video4",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video10",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


def lekiwi_cameras_config_realsense() -> dict[str, CameraConfig]:
    return {
        "top": RealSenseCameraConfig(
            serial_number_or_name="TOP_D455_SERIAL",
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist": RealSenseCameraConfig(
            serial_number_or_name="WRIST_CAMERA_SERIAL",
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    # 지금 OpenCV 테스트면 아래 유지
    return lekiwi_cameras_config_opencv()

    # 실제 depth까지 퍼블리시하려면 아래로 바꾸기
    # return lekiwi_cameras_config_realsense()


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    port: str = "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 0
    watchdog_timeout_ms: int = 0
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5