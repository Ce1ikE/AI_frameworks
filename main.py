from pathlib import Path
import pandas as pd

from pipeline_plugin.pipelines import *
from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(enable_logging=False)

train_dir = Path("./dataset/train")
test_dir = Path("./dataset/test")
input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.heic"))
input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.heic"))

def fetch_cluster_center():
    cluster_centers_file = Path("..\\py4MLP_pipelines\\results\\training_pipeline_20251110_051752\\Samples\\sample_003\\model_info.parquet")
    cluster_centers_df: pd.DataFrame = pd.read_parquet(cluster_centers_file) 
    kmeans_clusters = cluster_centers_df.iloc[0]["cluster_centers"]
    # adjust based on training results
    label_to_name = {
        10 : "alper",
        8  : "akif",
        0  : "arno",
        12 : "daiane",
        2  : "eh",
        4  : "lorenzo",
        6  : "rayen",
        5  : "robin",
        7  : "seppe",
        3  : "tj",
        1  : "thomas",
        9  : "ennis",
    }
    return {
        name: kmeans_clusters[idx]
        for idx, name in label_to_name.items()
    }

def compare_results():
    results_with_none = Path("..\\py4MLP_pipelines\\results\\inference_pipeline_20251111_220140\\sample_submission.csv")
    results_without_none = Path("..\\py4MLP_pipelines\\results\\inference_pipeline_20251111_211144\\sample_submission.csv")
    results_robin = Path("..\\py4MLP_pipelines\\results\\facedetectionRobin.csv")


    results_with_none_df = pd.read_csv(results_with_none)
    results_without_none_df = pd.read_csv(results_without_none)
    results_robin_df = pd.read_csv(results_robin)

    for label_EC_with_none , label_EC_without_none , label_Robin , image_name in zip(
        results_with_none_df["label_name"].values,
        results_without_none_df["label_name"].values,
        results_robin_df["label_name"].values,
        results_robin_df["image"].values,
    ):
        if label_EC_with_none != label_Robin or label_EC_without_none != label_Robin:
            print(f"labels for image {image_name}:")
            print(f"""
             {label_EC_with_none}
             {label_Robin}
             {label_EC_without_none}
            """)
        


def run_feature_extraction():
    feature_extraction_pipeline(input_files_train,core.paths.output)

def run_training():
    embedding_files = []
    training_pipeline(embedding_files,n_clusters=12,output_path=core.paths.output)

def run_inference():
    cluster_centers_and_labels = fetch_cluster_center()
    inference_pipeline(input_files_test,cluster_centers_and_labels,core.paths.output)

def run_streaming_inference():
    cluster_centers_and_labels = fetch_cluster_center()
    streaming_pipeline(cluster_centers_and_labels)

def run_slideshow():
    SlideShow.navigate_images([path for path in Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\inference_pipeline_20251111_220140").rglob("*.jpg") ])

def main():
    run_inference()

if __name__ == "__main__":
    main()

