from typing import List, Dict, Optional
import dask.dataframe as dd
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ======================================================
# Custom Exceptions
# ======================================================

class TraceAnalysisSchemaError(ValueError):
    """Raised when required columns are missing."""
    pass


class DaskTraceAnalysisError(RuntimeError):
    """Raised when distributed trace analysis fails."""
    pass


# ======================================================
# DaskTraceAnalyzer
# ======================================================

class DaskTraceAnalyzer:
    """
    Distributed Trace Clustering using Dask for large-scale event logs.

    This class handles trace construction and feature generation
    in a distributed manner and performs clustering on the reduced dataset.

    Design Principles
    -----------------
    • Dask for data-heavy preprocessing
    • Pandas + sklearn for algorithmic steps
    • API-safe (Django compatible)
    • Jupyter-friendly debugging

    Typical Usage
    -------------
    >>> analyzer = (
    ...     DaskTraceAnalyzer(ddf, n_clusters=5)
    ...         .validate_schema()
    ...         .build_traces()
    ...         .vectorize_traces()
    ...         .cluster_traces()
    ... )
    >>> analyzer.to_dataframe()
    """

    REQUIRED_COLUMNS = {
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    }

    # --------------------------------------------------
    def __init__(
        self,
        dask_df: dd.DataFrame,
        n_clusters: int = 5,
        normalize_vectors: bool = True,
        debug: bool = False
    ) -> None:
        """
        Initialize DaskTraceAnalyzer.

        Parameters
        ----------
        dask_df : dask.dataframe.DataFrame
            Cleaned large-scale event log.
        n_clusters : int, default=5
            Number of clusters for trace grouping.
        normalize_vectors : bool, default=True
            Whether to L2-normalize trace vectors.
        debug : bool, default=False
            Enable debug logging.
        """
        self.ddf = dask_df
        self.n_clusters = n_clusters
        self.normalize_vectors = normalize_vectors
        self.debug = debug

        self.traces: Optional[Dict[str, List[str]]] = None
        self.vector_df: Optional[pd.DataFrame] = None
        self.clustered_df: Optional[pd.DataFrame] = None

    # --------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[DaskTraceAnalyzer] {msg}")

    # --------------------------------------------------
    def validate_schema(self) -> "DaskTraceAnalyzer":
        """
        Check required event-log schema.

        Returns
        -------
        DaskTraceAnalyzer
        """
        missing = self.REQUIRED_COLUMNS - set(self.ddf.columns)
        if missing:
            raise TraceAnalysisSchemaError(
                f"Missing required columns: {sorted(missing)}"
            )
        self._log("Schema validation passed.")
        return self

    # --------------------------------------------------
    def build_traces(self) -> "DaskTraceAnalyzer":
        """
        Build ordered traces per case using Dask.

        Returns
        -------
        DaskTraceAnalyzer
        """
        try:
            self._log("Building traces using Dask...")

            def build_trace(df: pd.DataFrame) -> List[str]:
                return (
                    df.sort_values("time:timestamp")["concept:name"]
                    .astype(str)
                    .tolist()
                )

            traces = (
                self.ddf
                    .groupby("case:concept:name")
                    .apply(build_trace, meta=("trace", object))
                    .compute()
            )

            self.traces = traces.to_dict()
            self._log(f"{len(self.traces)} traces constructed.")
            return self

        except Exception as e:
            raise DaskTraceAnalysisError(
                f"Failed to build traces: {e}"
            ) from None

    # --------------------------------------------------
    def vectorize_traces(self) -> "DaskTraceAnalyzer":
        """
        Vectorize traces using Bag-of-Activities (BoA).

        Returns
        -------
        DaskTraceAnalyzer
        """
        try:
            if not self.traces:
                raise DaskTraceAnalysisError("No traces available.")

            self._log("Vectorizing traces (Bag-of-Activities)...")

            all_activities = sorted(
                {act for trace in self.traces.values() for act in trace}
            )

            vectors = []
            for case_id, trace in self.traces.items():
                counts = [trace.count(act) for act in all_activities]
                vectors.append([case_id] + counts)

            df = pd.DataFrame(
                vectors,
                columns=["case_id"] + all_activities
            ).set_index("case_id")

            if self.normalize_vectors:
                df[:] = normalize(df.values)

            self.vector_df = df
            self._log("Vectorization completed.")
            return self

        except Exception as e:
            raise DaskTraceAnalysisError(
                f"Trace vectorization failed: {e}"
            ) from None

    # --------------------------------------------------
    def cluster_traces(self) -> "DaskTraceAnalyzer":
        """
        Apply clustering algorithm on trace vectors.

        Returns
        -------
        DaskTraceAnalyzer
        """
        try:
            if self.vector_df is None:
                raise DaskTraceAnalysisError(
                    "Vector data missing. Call vectorize_traces() first."
                )

            self._log("Clustering traces with KMeans...")

            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init="auto"
            )

            labels = model.fit_predict(self.vector_df.values)

            clustered = self.vector_df.copy()
            clustered["cluster"] = labels

            self.clustered_df = clustered
            self._log("Clustering completed.")
            return self

        except Exception as e:
            raise DaskTraceAnalysisError(
                f"Clustering failed: {e}"
            ) from None

    # --------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export clustering results as DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        if self.clustered_df is None:
            raise DaskTraceAnalysisError("No clustering result available.")
        return self.clustered_df.reset_index()

    def to_dict(self) -> List[Dict]:
        """
        Export clustering results as JSON-serializable list.

        Returns
        -------
        list of dict
        """
        return self.to_dataframe().to_dict(orient="records")
