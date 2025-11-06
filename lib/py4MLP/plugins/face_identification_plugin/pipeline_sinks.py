from sklearn.metrics import silhouette_samples
from .dataclasses import *
from .transformers import *
from .utils.util_functions import *

import pandas as pd
import json
import plotly.express as px
from matplotlib import pyplot as plt
# https://www.geeksforgeeks.org/data-visualization/how-to-create-matplotlib-plots-without-a-gui/
import matplotlib
matplotlib.use('Agg')
plt.ioff()


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DataAggregator(PipelineSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [PipelineEventType.PIPELINE_BATCH_FINISHED, PipelineEventType.PIPELINE_SEQUENTIAL_FINISHED]

    def process(self, event: PipelineEventType):
        rows = []

        for sample_id, sample_info in self.pipeline_storage.sample_storage.items():
            sample_dir: Path = sample_info.sample_dir
            worker_storage: dict = sample_info.worker_storage

            embeddings: list = worker_storage.get(Keys.EMBEDDINGS_RECORDS, [])
            faces: list = worker_storage.get(Keys.FACE_RECORDS, [])
            
            for emb_path, face_path in zip(embeddings, faces):
                emb_full: Path = (sample_dir.parent / emb_path).resolve()
                face_full: Path = (sample_dir.parent / face_path).resolve()

                try:
                    emb_vector = np.load(emb_full)
                except Exception as e:
                    print(f"Could not load embedding {emb_full}: {e}")
                    continue

                rows.append({
                    "sample_id": sample_id,
                    "embedding": emb_vector.astype(np.float32),
                    "face_path": str(face_full)
                })
        df = pd.DataFrame(rows)
        self.pipeline_storage.pipeline_ctx[Keys.AGGREGATED_DF.value] = df
        print(f"DataAggregator built unified dataframe with {len(df)} rows.")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Reporter(PipelineSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [PipelineEventType.PIPELINE_FINISHED]

    def process(self, event: PipelineEventType):
        df: pd.DataFrame = self.pipeline_storage.pipeline_ctx[Keys.AGGREGATED_DF.value]
        if df is None:
            print("No aggregated data found — run DataAggregator first.")
            return

        time_diff = self.pipeline_storage.pipeline_end_time - self.pipeline_storage.pipeline_start_time
        time_diff_seconds = time_diff.total_seconds()
        report = {
            "total_images_with_faces": len(df["sample_id"].unique()),
            "total_embeddings": len(df["embedding"]),
            "embedding_dimension": df["embedding"].iloc[0].shape[0] if len(df) > 0 else 0,
            "pipeline": self.pipeline_storage.pipeline_composition,
            "start_time": str(self.pipeline_storage.pipeline_start_time),
            "end_time": str(self.pipeline_storage.pipeline_end_time),
            "duration (seconds)": str(time_diff_seconds),
        }

        report_path = self.pipeline_storage.pipeline_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"Report saved at {report_path}")

class PipelineReporter(PipelineSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [PipelineEventType.PIPELINE_FINISHED]

    def process(self, event: PipelineEventType):

        time_diff = self.pipeline_storage.pipeline_end_time - self.pipeline_storage.pipeline_start_time
        time_diff_seconds = time_diff.total_seconds()
        report = {
            "pipeline": self.pipeline_storage.pipeline_composition,
            "start_time": str(self.pipeline_storage.pipeline_start_time),
            "end_time": str(self.pipeline_storage.pipeline_end_time),
            "duration (seconds)": str(time_diff_seconds),
        }

        report_path = self.pipeline_storage.pipeline_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"Report saved at {report_path}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ParquetExporter(PipelineSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = PipelineEventType.PIPELINE_FINISHED

    def process(self, event: PipelineEventType):
        df: pd.DataFrame = self.pipeline_storage.pipeline_ctx[Keys.AGGREGATED_DF.value]
        if df is None:
            print("No aggregated data found — run DataAggregator first.")
            return

        path_to_file = self.pipeline_storage.pipeline_path / "embeddings.parquet"
        df.to_parquet(path_to_file, index=False)
        print(f"Parquet saved at {path_to_file}")