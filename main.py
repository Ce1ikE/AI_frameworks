#           _____                    _____                  
#          /\    \                  /\    \                 
#         /::\    \                /::\    \                
#        /::::\    \              /::::\    \               
#       /::::::\    \            /::::::\    \              
#      /:::/\:::\    \          /:::/\:::\    \             
#     /:::/__\:::\    \        /:::/  \:::\    \            
#    /::::\   \:::\    \      /:::/    \:::\    \           
#   /::::::\   \:::\    \    /:::/    / \:::\    \          
#  /:::/\:::\   \:::\    \  /:::/    /   \:::\    \         
# /:::/__\:::\   \:::\____\/:::/____/     \:::\____\        
# \:::\   \:::\   \::/    /\:::\    \      \::/    /        
#  \:::\   \:::\   \/____/  \:::\    \      \/____/         
#   \:::\   \:::\    \       \:::\    \                     
#    \:::\   \:::\____\       \:::\    \                    
#     \:::\   \::/    /        \:::\    \                   
#      \:::\   \/____/          \:::\    \                  
#       \:::\    \               \:::\    \                 
#        \:::\____\               \:::\____\                
#         \::/    /                \::/    /                
#          \/____/                  \/____/                 

from lib.Core import Core 
from lib.examples import (
    Example,
    detect_faces__pipeline, 
    detect_embed_faces__pipeline,
    detect_embed_classify_faces__pipeline,
    inference__pipeline,
    train_classifier__pipeline,
    convert_heic_to_jpg__pipeline,
    compile_all_results__pipeline,
)
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# main is the entrypoint of the application
# it sets up the PathManager for assuring that everything is in place, 
# Core sets up the necessary components like logging parsing config files and arguments 
# the Pipeline is where the actual work is done where the detector, embedder and reporter are used
# the detector detects faces in an image, the embedder creates embeddings for those faces
# the reporter saves the results to the output directory
# TODO: add a option to the reporter to save a PDF report
# TODO: add a option to the reporter to save a HTML report
# TODO: add a option to the reporter to save the visualization of the clustered embeddings
#       (TSNE: DONE, UMAP: TODO)
# TODO: add time measurements for the Pipeline 

core = Core(entrypoint=__file__)
logger = logging.getLogger(__name__)


test_dir = core.paths.input / "test"
train_dir = core.paths.input / "train"

input_files_to_convert = list(test_dir.glob("*.heic")) + list(train_dir.glob("*.heic"))
input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg"))
input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg"))

def main():
    logger.info("Starting main function")
    # choose which example to run    
    run_example = Example.DETECT_EMBED

    # ------------------ Example convert pipeline ------------------ #
    if run_example == Example.CONVERT:
        convert_heic_to_jpg__pipeline(
            input_files_to_convert, 
            delete_heic_files=False
        )

    # ------------------ Example detect pipeline ------------------ #
    elif run_example == Example.DETECT:
        detect_faces__pipeline(core).bulk_process(
            input_files_train, 
            continue_on_error=True
        )

    # ------------------ Example detect and embed pipeline ------------------ #
    elif run_example == Example.DETECT_EMBED:
        detect_embed_faces__pipeline(core).bulk_process(
            input_files_train, 
            continue_on_error=True
        )

    # ------------------ Example detect , embed and classify pipeline ------------------ #
    elif run_example == Example.DETECT_EMBED_CLASSIFY:
        cluster_centers_file = core.paths.output / "Training_KMeansClassifier" / "cluster_centers.parquet"
        df = pd.read_parquet(cluster_centers_file)
        if "cluster_centers" not in df.columns:
            raise ValueError(f"Cluster centers not found in {cluster_centers_file}")
        
        detect_embed_classify_faces__pipeline(core, df["cluster_centers"].to_numpy()).bulk_process(
            input_files_train,
            continue_on_error=True
        )

    # ------------------ Example compilation pipeline ------------------ #
    elif run_example == Example.COMPILE:
        compiled_results = core.paths.output / "Bulk_retinaface_mnet025_arcface_w600k_mbf_NoClassifier" / "compiled_results.parquet"
        path_to_dir = compiled_results.parent
        compile_all_results__pipeline(core,path=path_to_dir)

    # ------------------ Example training pipeline ------------------ #
    elif run_example == Example.TRAIN:
        train_classifier__pipeline(core).train(df, max_clusters=30)
    
    # ------------------ Example inference pipeline ------------------ #
    elif run_example == Example.INFERENCE:
        pass

if __name__ == "__main__":
    main()

