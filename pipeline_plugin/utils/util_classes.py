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
        # https://stackoverflow.com/questions/28595958/creating-trackbars-to-scroll-large-image-in-opencv-python/33293804#33293804
        state = {
            'zoom_factor': 1.0,
            'pan_x': 0, # Panning offset
            'pan_y': 0
        }

        def mouse_callback(event, x, y, flags, param):
            """
            Handles mouse wheel events for zooming.
            """
            
            # Zoom In: Mouse Wheel Up (Flags 8 or 16 depending on OS/backend)
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0: # Check for positive flags (usually wheel up)
                    # Zoom In: Increase factor by a small step
                    state['zoom_factor'] *= 1.1
                    if state['zoom_factor'] > 10.0: state['zoom_factor'] = 10.0 # Cap max zoom
                else:
                    # Zoom Out: Decrease factor by a small step
                    state['zoom_factor'] /= 1.1
                    if state['zoom_factor'] < 1.0: state['zoom_factor'] = 1.0 # Cap min zoom
                    
                # Reset pan if zooming out to the fit-to-screen size (zoom_factor 1.0)
                if state['zoom_factor'] == 1.0:
                    state['pan_x'] = 0
                    state['pan_y'] = 0

            if event == cv2.EVENT_LBUTTONDBLCLK:
                pass
            
        if len(image_paths) == 0:
            return
        
        import cv2
        
        i = 0
        window_name = "Display"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, mouse_callback)
        rect = cv2.getWindowImageRect(window_name)
        window_width = rect[2]
        window_height = rect[3]

        print(f"Fullscreen Window Dimensions: Width={window_width}, Height={window_height}")

        while True:
            img = cv2.imread(str(image_paths[i])) 
            if img is None:
                print(f"Error loading image: {image_paths[i].stem}")
                i = (i + 1) % len(image_paths)
                continue

            original_height, original_width = img.shape[:2]
            width_scale = window_width / original_width
            height_scale = window_height / original_height
            scale_factor = min(width_scale, height_scale)
            total_scale_factor = scale_factor * state['zoom_factor']
            # new dimensions
            new_width = int(original_width * total_scale_factor)
            new_height = int(original_height * total_scale_factor)

            resized_img = cv2.resize(img, (new_width, new_height),interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            
            if state['zoom_factor'] == 1.0:
                x_offset = (window_width - new_width) // 2
                y_offset = (window_height - new_height) // 2
            # If zoomed in, place the image at the top-left (pan_x/y offset could be added for panning)
            else:
                 # With simple zoom, we just keep the image in the top-left corner of the view
                 x_offset = (window_width - new_width) // 2
                 y_offset = (window_height - new_height) // 2
                
            # Use pan offsets (currently unused but provided for future panning implementation)
            x_offset += state['pan_x']
            y_offset += state['pan_y']

            c_x1 = max(0, x_offset)
            c_y1 = max(0, y_offset)
            c_x2 = min(window_width, x_offset + new_width)
            c_y2 = min(window_height, y_offset + new_height)

            # Define image region (clamped to image boundaries)
            # This handles cases where the image is partially off-screen
            i_x1 = max(0, -x_offset)
            i_y1 = max(0, -y_offset)
            i_x2 = min(new_width, window_width - x_offset)
            i_y2 = min(new_height, window_height - y_offset)
            
            canvas[c_y1:c_y2, c_x1:c_x2] = resized_img[i_y1:i_y2, i_x1:i_x2]
            
            cv2.setWindowTitle(window_name,image_paths[i].stem + f" (Zoom: {state['zoom_factor']:.2f}x)")
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n') or key == ord('p'):
                if key == ord('n'):
                    i = (i + 1) % len(image_paths)
                else:
                    i = (i - 1) % len(image_paths)
                
                state['zoom_factor'] = 1.0
                state['pan_x'] = 0
                state['pan_y'] = 0

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