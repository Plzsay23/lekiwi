from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

SERIAL = "333422301225"

config = RealSenseCameraConfig(
    serial_number_or_name=SERIAL,
    fps=30,
    width=640,
    height=480,
    use_depth=True,
)

camera = RealSenseCamera(config)
connected = False

try:
    camera.connect()
    connected = True

    color = camera.read()
    depth = camera.read_depth()

    print("color shape:", color.shape)
    print("depth shape:", depth.shape)

finally:
    if connected:
        camera.disconnect()