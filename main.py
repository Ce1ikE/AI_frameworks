from pathlib import Path
import matplotlib.pyplot as plt 
import pprint
import json
import pandas as pd

from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(enable_logging=False)

from pipeline_plugin.pipelines import *


train_dir = Path("./dataset/train")
input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.heic"))
test_dir = Path("./dataset/test")
input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.heic"))
embedding_files = [
    Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\feature_extraction_pipeline_20251110_025053\\processing_results.parquet")
]
def main():
    # feature_extraction_pipeline(input_files_train[:30],core.paths.output)

    training_pipeline(embedding_files,n_clusters=12,output_path=core.paths.output)
    
    # embedding_files = core.paths.output.glob("feature_extraction_pipeline*/*_cleaned.parquet")

    # embedding_files = core.paths.output.glob("feature_extraction_pipeline*/*.parquet")
    # filter(embedding_files)

    # df = pd.read_parquet("D:\\AI_frameworks\\py4MLP_pipelines\\results\\feature_extraction_pipeline_20251109_123233\\processing_results.parquet")
    # df.info()
    # img = Utils.decode_img(df[ExportKeys.FACE_IMAGE.value].iloc[0])
    # cv2.imshow("Display",img)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()

