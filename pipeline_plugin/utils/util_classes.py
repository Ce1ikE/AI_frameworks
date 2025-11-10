import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import json
from pathlib import Path
from ..dataclasses import ExportKeys

class Utils:
    def __init__( target_size=(160, 160)):
        target_size = target_size
    
    @staticmethod
    def resize(image, target_size=(160, 160)):
        return cv2.resize(image, target_size)

    @staticmethod
    def xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h])

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @staticmethod
    def validate_instance(obj, cls, name: str):
        if not isinstance(obj, cls):
            raise ValueError(f"{name} must be an instance of {cls.__name__}")
    
    @staticmethod
    def create_label_colors(labels):
        unique_labels = np.unique(labels)
        palette = sns.color_palette("tab20", len(unique_labels))
        label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
        colors = [label_to_color[lbl] for lbl in labels]
        handles = [
            plt.Line2D([], [], marker="o", color="w", markerfacecolor=palette[i], label=str(lbl), markersize=8)
            for i, lbl in enumerate(unique_labels)
        ]
        return colors, handles
    
    @staticmethod
    def encode_img(img: np.ndarray):
        success, encoded = cv2.imencode(".jpg", img)
        if success:
            return encoded.tobytes()
        else:
            return None 
        
    @staticmethod
    def decode_img(bytes):
        return cv2.imdecode(
            np.frombuffer(bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SlideShow:
    @staticmethod
    def navigate_images(image_paths: list[Path]):
        if len(image_paths) == 0:
            return

        import cv2
        i = 0
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        while True:
            img = cv2.imread(image_paths[i])
            # Show the image
            cv2.setWindowTitle("Display",image_paths[i].stem)
            cv2.imshow("Display", img)
            # Wait for action
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                i += 1
            if key == ord('p'):
                i -= 1

            if i < 0:
                i = len(image_paths) - 1
            if len(image_paths) == i:
                i = 0

        cv2.destroyAllWindows()

    @staticmethod
    def decode_images(results_path: Path):
        df = pd.read_parquet(results_path)
        df.info()

        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)

        for face_bytes in df[ExportKeys.FACE_IMAGE.value]:
            img = cv2.imdecode(
                np.frombuffer(face_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                print("decode error")
                continue

            cv2.imshow("Display",img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                pass

        cv2.destroyAllWindows()

    @staticmethod
    def filter_on_norm(embedding_file: Path,threshold: float):
        print(embedding_file)
        df = pd.read_parquet(embedding_file)
        df.info()
        input(f"Processing {embedding_file.stem} (continue)")
        # Assuming 'embedding' column contains lists or numpy arrays
        norms = df[ExportKeys.EMBEDDING_NORM.value]
        # Filter: keep only rows where norm >= 0.05
        df = df[norms >= threshold].reset_index(drop=True)
        print(f"Remaining rows after filtering: {len(df)}")
        df.info()
        input(f"finished Processing {embedding_file.stem} (continue)")