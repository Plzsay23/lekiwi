# !/usr/bin/env python

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class ArmOnlyDatasetConfig:
    repo_id: str = "<hf_username>/<dataset_repo_id>"
    single_task: str = "Pick up a floor egg and place it in the collection area."

    num_episodes: int = 1
    episode_time_s: float = 8.0
    reset_time_s: float = 1.0
    fps: int = 30

    root: str | Path | None = None
    use_videos: bool = True
    push_to_hub: bool = True
    image_writer_threads: int = 4


@dataclass
class LeKiwiArmOnlyRecordConfig:
    robot: LeKiwiClientConfig = field(default_factory=LeKiwiClientConfig)
    teleop: SO100LeaderConfig = field(default_factory=SO100LeaderConfig)
    dataset: ArmOnlyDatasetConfig = field(default_factory=ArmOnlyDatasetConfig)

    display_data: bool = True
    rerun_session_name: str = "lekiwi_arm_only_record"

    # LeKiwi gripper direction is often inverted relative to the leader.
    invert_gripper: bool = True


ARM_ACTION_KEYS = {
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
}

BASE_ZERO_ACTION = {
    "x.vel": 0.0,
    "y.vel": 0.0,
    "theta.vel": 0.0,
}


def arm_only_action_features(robot: LeKiwiClient) -> dict:
    return {
        key: value
        for key, value in robot.action_features.items()
        if key in ARM_ACTION_KEYS
    }


def arm_only_observation_features(robot: LeKiwiClient) -> dict:
    features = {}

    for key, value in robot.observation_features.items():
        # Arm joint state only
        if key in ARM_ACTION_KEYS:
            features[key] = value
            continue

        # Camera features are tuple-shaped, e.g. (H, W, C) or (H, W)
        if isinstance(value, tuple):
            features[key] = value
            continue

    return features


def filter_observation_for_dataset(observation: dict) -> dict:
    filtered = {}

    for key, value in observation.items():
        if key in ARM_ACTION_KEYS:
            filtered[key] = value
            continue

        # Keep camera images and depth arrays.
        # Exclude base velocity states: x.vel, y.vel, theta.vel.
        if key not in {"x.vel", "y.vel", "theta.vel"}:
            filtered[key] = value

    return filtered


def get_arm_action_from_leader(leader_arm: SO100Leader, invert_gripper: bool) -> dict:
    raw_action = leader_arm.get_action()

    if invert_gripper and "gripper.pos" in raw_action:
        raw_action["gripper.pos"] = 100.0 - raw_action["gripper.pos"]

    arm_action = {
        f"arm_{key}": float(value)
        for key, value in raw_action.items()
    }

    return {
        key: value
        for key, value in arm_action.items()
        if key in ARM_ACTION_KEYS
    }


def record_arm_only_loop(
    *,
    robot: LeKiwiClient,
    leader_arm: SO100Leader,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None,
    control_time_s: float,
    single_task: str,
    display_data: bool,
    invert_gripper: bool,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
):
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        loop_start_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()
        dataset_observation = filter_observation_for_dataset(observation)

        obs_processed = robot_observation_processor(dataset_observation)

        if dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features,
                obs_processed,
                prefix=OBS_STR,
            )

        arm_action = get_arm_action_from_leader(
            leader_arm=leader_arm,
            invert_gripper=invert_gripper,
        )

        # Dataset에는 arm action만 기록한다.
        action_values = teleop_action_processor((arm_action, observation))

        # 실제 LeKiwi host로 보낼 때는 base를 0으로 고정한다.
        # 이렇게 해야 같은 bus에 묶인 바퀴가 녹화 중 움직이지 않는다.
        robot_action_to_send = {
            **arm_action,
            **BASE_ZERO_ACTION,
        }
        robot_action_to_send = robot_action_processor(
            (robot_action_to_send, observation)
        )

        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(
                dataset.features,
                action_values,
                prefix=ACTION,
            )
            frame = {
                **observation_frame,
                **action_frame,
                "task": single_task,
            }
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
            )

        elapsed_s = time.perf_counter() - loop_start_t
        precise_sleep(max(1.0 / fps - elapsed_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


@draccus.wrap()
def main(cfg: LeKiwiArmOnlyRecordConfig):
    logging.basicConfig(level=logging.INFO)

    robot = LeKiwiClient(cfg.robot)
    leader_arm = SO100Leader(cfg.teleop)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(
        arm_only_action_features(robot),
        ACTION,
    )
    obs_features = hw_to_dataset_features(
        arm_only_observation_features(robot),
        OBS_STR,
    )
    dataset_features = {
        **action_features,
        **obs_features,
    }

    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        robot_type=robot.name,
        root=cfg.dataset.root,
        use_videos=cfg.dataset.use_videos,
        image_writer_threads=cfg.dataset.image_writer_threads,
    )

    robot.connect()
    leader_arm.connect()

    listener, events = init_keyboard_listener()

    if cfg.display_data:
        init_rerun(session_name=cfg.rerun_session_name)

    try:
        if not robot.is_connected or not leader_arm.is_connected:
            raise ValueError("Robot or leader arm is not connected.")

        recorded_episodes = 0

        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording arm-only episode {recorded_episodes}")

            record_arm_only_loop(
                robot=robot,
                leader_arm=leader_arm,
                events=events,
                fps=cfg.dataset.fps,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                invert_gripper=cfg.invert_gripper,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                recorded_episodes < cfg.dataset.num_episodes - 1
                or events["rerecord_episode"]
            ):
                log_say("Reset the environment")

                record_arm_only_loop(
                    robot=robot,
                    leader_arm=leader_arm,
                    events=events,
                    fps=cfg.dataset.fps,
                    dataset=None,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    invert_gripper=cfg.invert_gripper,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    finally:
        log_say("Stop recording")

        try:
            robot.send_action(BASE_ZERO_ACTION)
        except Exception:
            pass

        if robot.is_connected:
            robot.disconnect()

        if leader_arm.is_connected:
            leader_arm.disconnect()

        listener.stop()

        dataset.finalize()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub()