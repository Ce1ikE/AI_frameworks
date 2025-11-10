from pathlib import Path

from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(enable_logging=False)

from pipeline_plugin.pipelines import *


train_dir = Path("./dataset/train")
input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.heic"))
test_dir = Path("./dataset/test")
input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.heic"))
embedding_files = [
    Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\feature_extraction_pipeline_20251110_044330\\processing_results.parquet"),
    Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\feature_extraction_pipeline_20251110_044530\\processing_results.parquet"),
    Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\feature_extraction_pipeline_20251110_045016\\processing_results.parquet")
]

def main():
    feature_extraction_pipeline(input_files_train,core.paths.output)

    # embedding_file = Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\training_pipeline_20251110_043531\\Samples\\sample_002\\trained_embeddings.parquet")
    # df: pd.DataFrame = pd.read_parquet(embedding_file)    
    # embedding_norms = df[ExportKeys.EMBEDDING_NORM.value].values
    # embeddings = np.vstack(df[ExportKeys.EMBEDDING.value].values)
    # faces = np.vstack(df[ExportKeys.FACE_IMAGE.value].values)
    # PlotEmbeddings.norm_distribution(embedding_norms)
    # PlotEmbeddings.plot_embeddings_2d(embeddings,50,faces)
    
    # training_pipeline(embedding_files,n_clusters=12,output_path=core.paths.output)
    
    # if required drop cluster and create dictionary
    # cluster_centers = Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\training_pipeline_20251110_051752\\Samples\\sample_003\\model_info.parquet")
    # cluster_centers_df: pd.DataFrame = pd.read_parquet(cluster_centers) 
    # kmeans_clusters = cluster_centers_df.iloc[0]["cluster_centers"]
    # print(len(kmeans_clusters))
    # label_to_name = {

    # }
    # cluster_centers_and_labels = {
    #     name: kmeans_clusters[idx]
    #     for idx, name in label_to_name.items()
    # }

    # inference_pipeline(input_files_test,cluster_centers_and_labels,core.paths.output)

    # images = [path for path in Path("D:\\AI_frameworks\\py4MLP_pipelines\\results\\inference_pipeline_20251110_071947").rglob("*.jpg")]
    # SlideShow.navigate_images(images)

if __name__ == "__main__":
    main()

