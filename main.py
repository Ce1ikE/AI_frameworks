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

import argparse
from pathlib import Path
import pandas as pd

from pipeline_plugin.pipelines import *
from lib.py4MLP.py4MLP import Py4MLP


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Framework for face recognition pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("./dataset/train"),
        help="Directory containing training images"
    )
    
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("./dataset/test"),
        help="Directory containing test images"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: uses Py4MLP output path)"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=13,
        help="Number of clusters for training pipeline"
    )
    
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable Py4MLP logging"
    )
    
    parser.add_argument(
        "--run-feature-extraction",
        action="store_true",
        help="Run feature extraction pipeline"
    )
    
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Run training pipeline"
    )
    
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run inference pipeline"
    )
    
    parser.add_argument(
        "--run-streaming",
        action="store_true",
        help="Run streaming pipeline"
    )
    
    parser.add_argument(
        "--model-info",
        type=Path,
        default=Path("./pipeline_plugin/model_info.parquet"),
        help="Path to model info parquet file"
    )
    
    return parser.parse_args()


def get_image_files(directory: Path):
    """Get all image files from a directory."""
    supported_extensions = ["*.jpg", "*.png", "*.jpeg", "*.heic"]
    return [
        file for ext in supported_extensions 
        for file in directory.glob(ext)
    ]


def fetch_cluster_center(model_info_path: Path):
    """Fetch cluster centers and labels from model info file."""
    cluster_centers_df: pd.DataFrame = pd.read_parquet(model_info_path) 
    kmeans_clusters = cluster_centers_df.iloc[0]["cluster_centers"]
    
    # Adjust based on training results
    label_to_name = {
        10: "alper",
        8: "akif",
        0: "arno",
        12: "daiane",
        2: "eh",
        4: "lorenzo",
        6: "rayen",
        5: "robin",
        7: "seppe",
        3: "tj",
        1: "thomas",
        9: "ennis",
    }
    
    return {
        name: kmeans_clusters[idx]
        for idx, name in label_to_name.items()
    }


def main():
    """Main entry point for the AI framework pipeline."""
    args = parse_arguments()
    
    core = Py4MLP(enable_logging=args.enable_logging)
    output_dir = args.output_dir if args.output_dir else core.paths.output
    
    input_files_train = get_image_files(args.train_dir)
    input_files_test = get_image_files(args.test_dir)
    
    if args.run_feature_extraction:
        feature_extraction_pipeline(
            input_files_train,
            output_dir
        )
    
    # Training pipeline
    if args.run_training:
        embedding_files = []
        training_pipeline(
            embedding_files,
            n_clusters=args.n_clusters,
            output_path=output_dir,
            pipeline_name="training pipeline",
        )
    
    # Inference pipeline
    if args.run_inference:
        cluster_centers_and_labels = fetch_cluster_center(args.model_info)
        inference_pipeline(
            input_files_test,
            cluster_centers_and_labels,
            output_dir
        )
    
    # Streaming pipeline
    if args.run_streaming:
        cluster_centers_and_labels = fetch_cluster_center(args.model_info)
        streaming_pipeline(cluster_centers_and_labels)

if __name__ == "__main__":
    main()

