#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


@dataclass
class LeKiwiServerConfig:
    """Configuration for the LeKiwi host script."""

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

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring LeKiwi")
    robot = LeKiwi(cfg.robot)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host = LeKiwiHost(cfg.host)

    last_cmd_time = time.time()
    logging.info("Waiting for commands...")

    start = time.perf_counter()

    try:
        while True:
            if host.connection_time_s > 0:
                duration = time.perf_counter() - start
                if duration >= host.connection_time_s:
                    logging.info("Connection time reached. Exiting host loop.")
                    break

            loop_start_time = time.time()

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

                last_observation = robot.get_observation()

                for cam_key in robot.cameras:
                    if cam_key not in last_observation:
                        continue
                    ret, buffer = cv2.imencode(
                        ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ret:
                        last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        last_observation[cam_key] = ""

                try:
                    host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as e:
                    logging.warning("Observation send failed: %s", e)

            except Exception as e:
                logging.exception("Main host loop error: %s", e)
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

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
