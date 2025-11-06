import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import json

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
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)