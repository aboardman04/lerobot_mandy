import pathlib

path = pathlib.Path("usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0")
by_id = pathlib.Path("/dev/v4l/by-id") / path
print("By id exists?", by_id.exists())
