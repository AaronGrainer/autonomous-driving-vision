import cv2
import typer

from scripts.autonomous_detector import AutonomousDetector
from src.road_detection.auto_drive.detect import RoadDetector

app = typer.Typer()


@app.command()
def road_detection(detect_type: str = "image"):
    if detect_type == "image":
        img_path = "dataset/road.png"
        img = cv2.imread(img_path)[:, :, ::-1]

        road_detector = RoadDetector()
        road_detector.detect_img(img)
    elif detect_type == "video":
        input_video = "test_asset/usa_laguna.mp4"

        road_detector = RoadDetector()
        road_detector.detect_video(input_video)


@app.command()
def full_predict(detect_type="image", frame_skip: int = None):
    if detect_type == "image":
        img_path = "data/test_asset/gettyimages-1134565289-640x640.jpeg"
        img = cv2.imread(img_path)[:, :, ::-1]

        autonomous_detector = AutonomousDetector()
        autonomous_detector.detect_img(img)
    elif detect_type == "video":
        input_video = "data/test_asset/usa_laguna.mp4"

        autonomous_detector = AutonomousDetector()
        autonomous_detector.detect_video(input_video, frame_skip)
    else:
        raise ValueError


if __name__ == "__main__":
    app()
