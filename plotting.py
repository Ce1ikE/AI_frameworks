from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
from pipeline_plugin.dataclasses import *
from pipeline_plugin.hybrid_sinks import DimensionalityVisualizer
from pipeline_plugin.worker_sinks import TrainingEvaluator
from pipeline_plugin.utils.util_classes import Utils 
matplotlib.use('Agg')
plt.ioff()

FONTDICT = {
    "fontsize": 10,
    "fontweight": "bold",
    "fontfamily": "monospace",
}

def plot_results(df: pd.DataFrame, save_dir: Path):
    
    embeddings = np.vstack(df[ExportKeys.EMBEDDING_NORMALIZED.value].values)
    labels = df["KMeans"].values
    label_colors, handles = Utils.create_label_colors(labels)

    dv = DimensionalityVisualizer(name="dimensionality visualizer", method=DimensionalityVisualizer.ReductionTechnique.TSNE)
    dv.save_dir = save_dir
    projections_2d = dv._compute_projection(
        embeddings,
        dims=2,
        method=dv.method
    )
    projections_3d = dv._compute_projection(
        embeddings,
        dims=3,
        method=dv.method
    )
    dv._plot_2d(
        projections_2d,
        label_colors, 
        handles,
        "classified_normalized",
        dv.method
    )
    dv._plot_3d(
        projections_3d,
        label_colors, 
        "classified_normalized",
        dv.method
    )
    dv._plot_2d_visualisation_with_thumbnails(
        projections_2d,
        df,
        label_colors,
        "classified_normalized",
        dv.method
    )

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(
        projections_3d[:,0],
        projections_3d[:,1],
        projections_3d[:,2],
        s=12,
        c=label_colors,
        alpha=0.9,
        marker="o",
        edgecolors="white",
        linewidths=0.2
    )
    ax.set_title(f"{dv.method.value.upper()} Visualization (3D) normalized", fontdict=FONTDICT)
    ax.set_xlabel(f"{dv.method.value.upper()} Dim 1", fontdict=FONTDICT)
    ax.set_ylabel(f"{dv.method.value.upper()} Dim 2", fontdict=FONTDICT)
    ax.set_zlabel(f"{dv.method.value.upper()} Dim 3", fontdict=FONTDICT)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_dir / f"{dv.method.value.lower()}_3d_classified_normalized.svg", format="svg")
    plt.close()

    te = TrainingEvaluator("training evaluator")
    te.sample_dir = save_dir
    (te.sample_dir / "KMeans").mkdir(parents=True, exist_ok=True)
    te._plot_silhouette_analysis(
        embeddings,
        labels,
        model_name="KMeans",
    )



results_retinaface_mnet025_arcface_mbf = Path(
    r"..\py4MLP_pipelines\results\training_pipeline_retinaface_mnet025_arcface__mbf_20251215_231406_20251216_034059\Samples\sample_001"
)
results_retinaface_mnet025_arcface_r50 = Path(
    r"..\py4MLP_pipelines\results\training_pipeline_retinaface_mnet025_arcface__r50_20251215_231732_20251216_034149\Samples\sample_001"
)
results_retinaface_r34_arcface_mbf = Path(
    r"..\py4MLP_pipelines\results\training_pipeline_retinaface_r34_arcface__mbf_20251215_231956_20251216_034254\Samples\sample_001"
)
results_retinaface_r34_arcface_r50 = Path(
    r"..\py4MLP_pipelines\results\training_pipeline_retinaface_r34_arcface__r50_20251215_232249_20251216_034352\Samples\sample_001"
)

def main():
    for results_dir in [
        results_retinaface_mnet025_arcface_mbf,
        results_retinaface_mnet025_arcface_r50,
        results_retinaface_r34_arcface_mbf,
        results_retinaface_r34_arcface_r50,
    ]:
        df = pd.read_parquet(
            results_dir / "trained_embeddings.parquet"
        )

        # df.info()

        plot_results(
            df,
            results_dir.parent.parent,
        )


if __name__ == "__main__":
    main()
