import pandas as pd
import numpy as np
from typing import Optional, Literal
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


class TraceAnalysisError(RuntimeError):
    """
    Raised when trace-level analysis fails due to invalid input,
    missing columns, or internal processing errors.
    """
    pass


class TraceAnalyzer:
    """
    Perform trace-level analytical operations on an event log.

    This class is responsible for:
    - Extracting activity sequences (traces) per case
    - Vectorizing traces using Bag of Activities (BoA)
    - Applying clustering algorithms (KMeans)
    - Attaching cluster labels back to the event-level log

    Designed for API usage in Django (service layer),
    and fully compatible with LogPreprocessor output.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp"
    ) -> None:
        """
        Initialize the TraceAnalyzer with a cleaned event log DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Event-level DataFrame produced after preprocessing.
        case_col : str, default="case:concept:name"
            Column name representing case identifiers.
        activity_col : str, default="concept:name"
            Column name representing activities.
        timestamp_col : str, default="time:timestamp"
            Column name representing event timestamps.

        Raises
        ------
        TraceAnalysisError
            If the input DataFrame is invalid or required columns are missing.
        """
        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise TraceAnalysisError("Input df must be a valid pandas DataFrame.")

            self.df = df.copy()
            self.case_col = case_col
            self.activity_col = activity_col
            self.timestamp_col = timestamp_col

            self.traces: Optional[pd.Series] = None
            self.vectorizer: Optional[CountVectorizer] = None
            self.trace_vectors: Optional[np.ndarray] = None
            self.cluster_labels: Optional[np.ndarray] = None

            self._validate_required_columns()

        except Exception as e:
            raise TraceAnalysisError(f"Failed to initialize TraceAnalyzer: {e}") from None

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------
    def _validate_required_columns(self) -> None:
        """
        Validate presence of required columns in the DataFrame.

        Raises
        ------
        TraceAnalysisError
            If any required column is missing.
        """
        required = {self.case_col, self.activity_col}
        missing = required - set(self.df.columns)

        if missing:
            raise TraceAnalysisError(
                f"Missing required columns for trace analysis: {missing}"
            )

    # ------------------------------------------------------------------
    # Step 1: Trace extraction
    # ------------------------------------------------------------------
    def build_traces(self) -> pd.Series:
        """
        Extract ordered activity sequences (traces) for each case.

        Events are sorted by timestamp if the timestamp column exists.

        Returns
        -------
        pd.Series
            Index: case_id
            Value: list of activity names forming the trace

        Raises
        ------
        TraceAnalysisError
            If trace extraction fails.
        """
        try:
            df = self.df.copy()

            if self.timestamp_col in df.columns:
                df = df.sort_values(self.timestamp_col)

            self.traces = (
                df.groupby(self.case_col)[self.activity_col]
                  .apply(list)
            )

            return self.traces

        except Exception as e:
            raise TraceAnalysisError(f"Failed to build traces: {e}") from None

    # ------------------------------------------------------------------
    # Step 2: Trace vectorization
    # ------------------------------------------------------------------
    def vectorize_traces(
        self,
        method: Literal["boa"] = "boa"
    ) -> np.ndarray:
        """
        Convert traces into numeric vectors suitable for clustering.

        Currently supported methods:
        - 'boa': Bag of Activities

        Parameters
        ----------
        method : {'boa'}, default='boa'
            Vectorization method.

        Returns
        -------
        np.ndarray
            Numeric trace vectors (n_traces × n_features).

        Raises
        ------
        TraceAnalysisError
            If vectorization fails or an unsupported method is requested.
        """
        try:
            if self.traces is None:
                self.build_traces()

            if method != "boa":
                raise TraceAnalysisError(
                    f"Unsupported vectorization method '{method}'."
                )

            sentences = self.traces.apply(lambda acts: " ".join(acts))

            self.vectorizer = CountVectorizer()
            self.trace_vectors = (
                self.vectorizer
                .fit_transform(sentences)
                .toarray()
            )

            return self.trace_vectors

        except Exception as e:
            raise TraceAnalysisError(f"Failed to vectorize traces: {e}") from None

    # ------------------------------------------------------------------
    # Step 3: Trace clustering
    # ------------------------------------------------------------------
    def cluster_traces(
        self,
        n_clusters: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Cluster traces using the KMeans algorithm.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of clusters to generate.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ['case:concept:name', 'cluster_id']

        Raises
        ------
        TraceAnalysisError
            If clustering fails or input data is invalid.
        """
        try:
            if self.trace_vectors is None:
                self.vectorize_traces()

            if n_clusters <= 1:
                raise TraceAnalysisError("n_clusters must be greater than 1.")

            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto"
            )

            self.cluster_labels = model.fit_predict(self.trace_vectors)

            return pd.DataFrame({
                self.case_col: self.traces.index,
                "cluster_id": self.cluster_labels
            })

        except Exception as e:
            raise TraceAnalysisError(f"Trace clustering failed: {e}") from None

    # ------------------------------------------------------------------
    # Step 4: Attach clustering result to event log
    # ------------------------------------------------------------------
    def attach_clusters_to_log(self) -> pd.DataFrame:
        """
        Attach cluster labels to the original event-level DataFrame.

        Returns
        -------
        pd.DataFrame
            Event-level DataFrame with an additional 'cluster_id' column.

        Raises
        ------
        TraceAnalysisError
            If clustering has not been executed yet.
        """
        try:
            if self.cluster_labels is None or self.traces is None:
                raise TraceAnalysisError(
                    "Clustering has not been performed yet."
                )

            case_cluster_map = dict(
                zip(self.traces.index, self.cluster_labels)
            )

            df_out = self.df.copy()
            df_out["cluster_id"] = (
                df_out[self.case_col]
                .map(case_cluster_map)
            )

            return df_out

        except Exception as e:
            raise TraceAnalysisError(
                f"Failed to attach cluster labels to log: {e}"
            ) from None
