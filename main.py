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
from lib.API.Preprocessor import Preprocessor
from lib.examples import (
    advanced_example_with_direct_config__pipeline,
    detect_and_embed_faces__pipeline,
    detect_embed_and_classify_faces__pipeline,
    detect_faces__pipeline, 
    train_classifier__pipeline,
    convert_heic_to_jpg__pipeline,
)
from pathlib import Path
import pprint
# main is the entrypoint of the application
# it sets up the PathManager for assuring that everything is in place, 
# Core sets up the necessary components like logging parsing config files and arguments 
# the Pipeline is where the actual work is done where the detector, embedder and reporter are used
# the detector detects faces in an image, the embedder creates embeddings for those faces
# the reporter saves the results to the output directory
# TODO: add a ML model for classification or clustering of the embeddings
# TODO: add a option to the reporter to save a PDF report
# TODO: add a option to the reporter to save a HTML report
# TODO: add training of a classifier on the embeddings to the Pipeline
# TODO: add a option to the reporter to save the classifier model
# TODO: add a module to load a classifier model and use it in the Pipeline
# TODO: add a option to the reporter to save the visualization of the embeddings
# TODO: add a option to the reporter to save the visualization of the clusters
# TODO: add time measurements to the Pipeline and reporter

core = Core(entrypoint=__file__)

def main():
    core.logger.info("Starting main function")

    # first we don't want any HEIC files, so we convert them to JPG
    input_files = list(core.paths.input.glob("*.heic"))
    core.logger.info(f"Found {len(input_files)} input files")
    if len(input_files) > 0:
        convert_heic_to_jpg__pipeline(core, input_files, delete_heic_files=True)
    
    # if everything is JPG or PNG now, we can proceed with the other pipelines
    input_files = list(core.paths.input.glob("*.jpg")) + list(core.paths.input.glob("*.png")) + list(core.paths.input.glob("*.jpeg"))
    core.logger.info(f"Found {len(input_files)} input files")
    detect_and_embed_faces__pipeline(core).run(input_files)
    
    
if __name__ == "__main__":
    main()

