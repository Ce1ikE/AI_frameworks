# TODO	                                
# ----
# [x] Save input images	                    
# [x] Save cropped faces	                
# [x] Save annotated image	                
# [x] Save embedding vectors	            
# [X] Save model settings	                	            
# [X] Save trained model (ONNX)	            	            
# [x] Compile results         	            
# [x] Save clustered embeddings	            
# [X] Save classification labels (inference pipeline required)	        
# [X] Save evaluation report (how good is the data ?)	                	            
# [X] Save training report (what models used)	                	            
# [X] Save inference report (what models used and time metrics)
# [ ] Save ROC curve for different confidence thresholds (requires labeled data though)	                    
# [x] Save UMAP visualization	            
# [x] Save silhouette scores
# [x] Save TSNE visualization
# [x] Save UMAP and TSNE combined visualization
# [x] Save silhouette analysis plot	            
# [x] Save elbow method plot	            
# [ ] Add autolabel class (like in Deepbee to correct model's prediction) (input: dict of possibilities (name1,name2,etc...,others)) !!!

from pathlib import Path
import pandas as pd

from pipeline_plugin.pipelines import *
from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(enable_logging=False)

train_dir = Path("./dataset/train")
test_dir = Path("./dataset/test")

input_files_train = [
    file for ext in ["*.jpg", "*.png", "*.jpeg", "*.heic"] 
    for file in train_dir.glob(ext)
]
input_files_test = [
    file for ext in ["*.jpg", "*.png", "*.jpeg", "*.heic"] 
    for file in test_dir.glob(ext)
]

def fetch_cluster_center():
    cluster_centers_file = Path("./pipeline_plugin/model_info.parquet")
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

def main():
    # feature_extraction_pipeline(
    #   input_files_train,
    #   core.paths.output
    # )
   
    # embedding_files = []
    # training_pipeline(
    #     embedding_files,
    #     n_clusters=13,
    #     output_path=core.paths.output,
    #     pipeline_name=f"training pipeline",
    # )

    # cluster_centers_and_labels = fetch_cluster_center()
    # inference_pipeline(input_files_test,cluster_centers_and_labels,core.paths.output)

    cluster_centers_and_labels = fetch_cluster_center()
    streaming_pipeline(cluster_centers_and_labels)

if __name__ == "__main__":
    main()

