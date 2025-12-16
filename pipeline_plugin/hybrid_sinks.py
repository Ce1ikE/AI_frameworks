from .dataclasses import *
from .transformers import *
from .utils.util_classes import *

from enum import Enum
import plotly.express as px

class DimensionalityVisualizer(PipelineSink,WorkerSink):
    class ReductionTechnique(Enum):
        TSNE = "TSNE" 
        UMAP = "UMAP" 
    
    def __init__(
        self, 
        name: str,
        method: ReductionTechnique = ReductionTechnique.TSNE,
        random_state: int = 42,
        perplexity: int = 30,
        n_iterations: int = 1000,
    ):
        super().__init__(name)
        self.method = method
        self.random_state = random_state
        self.perplexity = perplexity
        self.n_iterations = n_iterations

        self.input_type = [
            PipelineEventType.PIPELINE_FINISHED,
            TrainingResults
        ]

        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def process(self, data: PipelineEventType | TrainingResults):

        if isinstance(data,PipelineEventType):
            self.save_dir = self.pipeline_storage.pipeline_path

            for df_path in self.pipeline_storage.pipeline_ctx.get(PipelineKeys.AGGREGATED_RECORDS, []):
                df: pd.DataFrame = pd.read_parquet(df_path)
                df.info()
                embeddings = np.vstack(df[ExportKeys.EMBEDDING_NORMALIZED.value].values)
                projections_2d = self._compute_projection(embeddings,dims=2,method=self.method)
                projections_3d = self._compute_projection(embeddings,dims=3,method=self.method)

                self._plot_2d(
                    projections_2d,
                    None, 
                    None,
                    "",
                    self.method
                )
                self._plot_3d(
                    projections_3d,
                    None, 
                    "",
                    self.method
                )
                self._plot_2d_visualisation_with_thumbnails(
                    projections_2d,
                    df,
                    None, 
                    "",
                    self.method
                )

        if isinstance(data,TrainingResults):
            df = data.embeddings.embeddings
            df.info()
            embeddings = np.vstack(df[ExportKeys.EMBEDDING_NORMALIZED.value])
            projections_2d = self._compute_projection(embeddings,dims=2,method=self.method)
            projections_3d = self._compute_projection(embeddings,dims=3,method=self.method)

            for trained_model in data.models:
                label_column = trained_model.model_name
                if label_column in df:
                    labels = df[label_column]

                self.save_dir = self.sample_dir / label_column
                self.save_dir.mkdir(parents=True,exist_ok=True)
                label_colors, handles = Utils.create_label_colors(labels)
                self._plot_2d(
                    projections_2d,
                    label_colors, 
                    handles,
                    f"({label_column})",
                    self.method
                )
                self._plot_3d(
                    projections_3d,
                    label_colors, 
                    f"({label_column})",
                    self.method
                )
                self._plot_2d_visualisation_with_thumbnails(
                    projections_2d,
                    df,
                    label_colors, 
                    f"({label_column})",
                    self.method
                )

    def _compute_projection(self, embeddings, dims=2, method=ReductionTechnique.TSNE):
        if method == self.ReductionTechnique.TSNE:
            from sklearn.manifold import TSNE
            if len(embeddings) <= self.perplexity:
                raise ValueError("TSNE: perplexity is higher than number of samples")
            tsne = TSNE(
                n_components=dims,
                perplexity=self.perplexity,
                max_iter=self.n_iterations,
                random_state=self.random_state
            )
            return tsne.fit_transform(embeddings)
        
        elif method == self.ReductionTechnique.UMAP:
            import umap
            reducer = umap.UMAP(n_components=dims, random_state=self.random_state)
            return reducer.fit_transform(embeddings)
        
        else:
            raise ValueError(f"unknown projection method: {method}")

    def _plot_2d(
        self,
        projections,
        label_colors,
        handles,
        suffix,
        method: ReductionTechnique
    ):
        plt.figure(figsize=(10,10))
        plt.scatter(
            projections[:,0],
            projections[:,1],
            s=12,
            c=label_colors,
            alpha=0.9,
            marker="o",
            edgecolors="white",
            linewidths=0.2
        )
        if handles:
            plt.legend(handles=handles, title="Labels")

        plt.title(f"{method.value.upper()} Visualization (2D) {suffix}", fontdict=self.fontdict)
        plt.xlabel(f"{method.value.upper()} Dim 1", fontdict=self.fontdict)
        plt.ylabel(f"{method.value.upper()} Dim 2", fontdict=self.fontdict)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{method.value.lower()}_2d_{suffix}.svg")
        plt.close()

    def _plot_3d(
        self, 
        projections, 
        labels, 
        suffix: str, 
        method: ReductionTechnique
    ):
        fig = px.scatter_3d(
            x=projections[:,0],
            y=projections[:,1],
            z=projections[:,2],
            color=labels,
            title=f"{method.value.upper()} Visualization (3D) {suffix}"
        )
        fig.update_traces(
            marker={
                "size": 2,
                "opacity": 0.8,
            }
        )
        fig.write_html(self.save_dir / f"{method.value.lower()}_3d_{suffix}.html")

    def _plot_2d_visualisation_with_thumbnails(
        self,
        projection_2d,
        df,
        label_colors,
        suffix: str, 
        method: ReductionTechnique
    ):
        # if face images are provided, we'll use them as markers in custom plot
        # https://learnopencv.com/t-sne-for-feature-visualization/
        from PIL import Image, ImageOps

        faces = df[ExportKeys.FACE_IMAGE.value]
        # determine the range for x and y axes to properly place images
        x_min, x_max = projection_2d[:, 0].min(), projection_2d[:, 0].max()
        y_min, y_max = projection_2d[:, 1].min(), projection_2d[:, 1].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        # each thumbnail is 3% of the span
        thumb_frac = 0.05  
        thumb_w = x_span * thumb_frac
        thumb_h = y_span * thumb_frac
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()

        if label_colors is None:
            colors = [ 0 for _ in range(len(projection_2d))]
        else:
            colors = label_colors 

        plt.title(f"{method.value.upper()} Visualization with Face Thumbnails {suffix}", fontdict=self.fontdict)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axis("off")
        
        for (x, y), img_bytes, color in zip(projection_2d, faces,colors):
            if img_bytes is not None:
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
        plt.savefig(self.save_dir / f"{method.value.lower()}_2d_faces_{suffix}.svg", format="svg")
        plt.close()