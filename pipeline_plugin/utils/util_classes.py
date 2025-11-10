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


class PlotEmbeddings:
    fontdict = {
        "fontsize": 10,
        "fontweight": "bold",
        "fontfamily": "monospace",
    }

    @classmethod
    def norm_distribution(cls,embedding_norms):
        
        fig = plt.figure(figsize=(10, 6))
        plt.hist(embedding_norms, bins=100)
        plt.title("Embedding Norm Distribution",  fontdict=cls.fontdict)
        plt.xlabel("Frequency",  fontdict=cls.fontdict)
        plt.ylabel("L2 Norm",  fontdict=cls.fontdict)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("norm_distr.svg",format="svg")
        plt.close(fig)

    @classmethod
    def plot_embeddings_2d(cls,embeddings,perplexity,face_images):
        
        from sklearn.manifold import TSNE
        if len(embeddings) <= perplexity:
            raise ValueError("TSNE: perplexity is higher than number of samples")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42
        )
        projections = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10,10))
        plt.scatter(
            projections[:,0],
            projections[:,1],
            s=12,
            c=None,
            alpha=0.9,
            marker="o",
            edgecolors="white",
            linewidths=0.2
        )
        plt.title(f"Visualization (2D)", fontdict=cls.fontdict)
        plt.xlabel(f"Dim 1", fontdict=cls.fontdict)
        plt.ylabel(f"Dim 2", fontdict=cls.fontdict)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("tsne_Visualization.svg",format="svg")
        plt.close()

        from PIL import Image, ImageOps
        label_colors = None
        faces = face_images
        # determine the range for x and y axes to properly place images
        x_min, x_max = projections[:, 0].min(), projections[:, 0].max()
        y_min, y_max = projections[:, 1].min(), projections[:, 1].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        # each thumbnail is 3% of the span
        thumb_frac = 0.05  
        thumb_w = x_span * thumb_frac
        thumb_h = y_span * thumb_frac
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        if label_colors is None:
            colors = [ 0 for _ in range(len(projections))]
        else:
            colors = label_colors 

        plt.title(f"Visualization with Face Thumbnails", fontdict=cls.fontdict)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axis("off")
        
        for (x, y), img_bytes, color in zip(projections, faces,colors):
            img = Utils.decode_img(img_bytes)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((40, 40), Image.Resampling.LANCZOS)

            if label_colors is not None:
                img = img.convert("RGBA")
                new_color = tuple([int(channel*255) for channel in color])
                img = ImageOps.expand(
                    img, 
                    border=5, 
                    fill=new_color
                )

            ax.imshow(
                img,
                extent=(x - thumb_w/2, x + thumb_w/2, y - thumb_h/2, y + thumb_h/2),
                zorder=2,
                alpha=0.9
            )

        plt.tight_layout()
        plt.savefig("tsne_2d_faces.svg", format="svg")
        plt.close()