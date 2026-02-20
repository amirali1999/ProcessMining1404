from typing import List, Dict, Optional
import dask.dataframe as dd
import pandas as pd

from .pattern_miner import PatternMiner


# ======================================================
# Custom Exceptions
# ======================================================

class PatternMiningSchemaError(ValueError):
    """Raised when required columns are missing in the event log."""
    pass


class DaskPatternMiningError(RuntimeError):
    """Raised when distributed pattern mining fails."""
    pass


# ======================================================
# DaskPatternMiner
# ======================================================

class DaskPatternMiner:
    """
    Distributed frequent pattern mining orchestrator for very large logs.

    This class handles **scalable trace construction** using Dask and
    delegates the core mining algorithm to `PatternMiner`.

    Design Principles
    -----------------
    • Distributed data preparation (Dask)
    • Centralized algorithmic logic (PatternMiner)
    • API-safe for Django services
    • Jupyter-friendly introspection

    Typical Usage
    -------------
    >>> miner = (
    ...     DaskPatternMiner(ddf, min_support=100)
    ...         .validate_schema()
    ...         .build_traces()
    ...         .mine_patterns()
    ... )
    >>> miner.to_dataframe()
    """

    REQUIRED_COLUMNS = {
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    }

    def __init__(
        self,
        dask_df: dd.DataFrame,
        min_support: int,
        debug: bool = False
    ) -> None:
        """
        Initialize DaskPatternMiner.

        Parameters
        ----------
        dask_df : dask.dataframe.DataFrame
            Preprocessed event log (output of LargeLogPreprocessor).
        min_support : int
            Minimum support threshold for frequent patterns.
        debug : bool, default=False
            Enables debug logging (development / Jupyter only).
        """
        self.ddf = dask_df
        self.min_support = min_support
        self.debug = debug

        self.traces: Optional[List[List[str]]] = None
        self.patterns: Optional[List[Dict]] = None

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[DaskPatternMiner] {msg}")

    # --------------------------------------------------
    # Schema validation
    # --------------------------------------------------
    def validate_schema(self) -> "DaskPatternMiner":
        """
        Validate required event log schema.

        Ensures essential process mining columns exist.

        Returns
        -------
        DaskPatternMiner

        Raises
        ------
        PatternMiningSchemaError
            If required columns are missing.
        """
        try:
            missing = self.REQUIRED_COLUMNS - set(self.ddf.columns)
            if missing:
                raise PatternMiningSchemaError(
                    f"Missing required columns: {sorted(missing)}"
                )

            self._log("Schema validation passed.")
            return self

        except PatternMiningSchemaError:
            raise

        except Exception as e:
            raise DaskPatternMiningError(
                f"Schema validation failed: {e}"
            ) from None

    # --------------------------------------------------
    # Trace construction
    # --------------------------------------------------
    def build_traces(self) -> "DaskPatternMiner":
        """
        Build ordered traces from distributed event log.

        Groups events by case and sorts them temporally.
        The resulting trace list is computed and stored locally.

        Returns
        -------
        DaskPatternMiner

        Raises
        ------
        DaskPatternMiningError
            If trace construction fails.
        """
        try:
            self._log("Building traces using Dask...")

            def build_case_trace(df: pd.DataFrame) -> List[str]:
                return (
                    df.sort_values("time:timestamp")["concept:name"]
                    .astype(str)
                    .tolist()
                )

            traces_series = (
                self.ddf
                    .groupby("case:concept:name")
                    .apply(build_case_trace, meta=("trace", object))
                    .compute()
            )

            self.traces = traces_series.tolist()
            self._log(f"{len(self.traces)} traces constructed.")

            return self

        except Exception as e:
            raise DaskPatternMiningError(
                f"Failed to build traces: {e}"
            ) from None

    # --------------------------------------------------
    # Mining delegation
    # --------------------------------------------------
    def mine_patterns(self) -> "DaskPatternMiner":
        """
        Mine frequent sequential patterns using PatternMiner.

        Delegates algorithmic work to the core miner.

        Returns
        -------
        DaskPatternMiner

        Raises
        ------
        DaskPatternMiningError
            If mining fails.
        """
        try:
            if not self.traces:
                raise DaskPatternMiningError(
                    "Traces are empty. Call build_traces() first."
                )

            self._log("Delegating pattern mining to PatternMiner...")

            miner = PatternMiner(
                traces=self.traces,
                min_support=self.min_support
            )

            self.patterns = miner.mine()
            self._log(f"{len(self.patterns)} patterns mined.")

            return self

        except Exception as e:
            raise DaskPatternMiningError(
                f"Pattern mining failed: {e}"
            ) from None

    # --------------------------------------------------
    # Output helpers
    # --------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export mined patterns as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ['pattern', 'support']

        Raises
        ------
        DaskPatternMiningError
            If patterns are unavailable.
        """
        if self.patterns is None:
            raise DaskPatternMiningError(
                "No patterns available. Call mine_patterns() first."
            )

        return pd.DataFrame(self.patterns)

    def to_dict(self) -> List[Dict]:
        """
        Export mined patterns as a list of dictionaries
        (JSON-serializable, API-ready).

        Returns
        -------
        list of dict
        """
        if self.patterns is None:
            raise DaskPatternMiningError(
                "No patterns available. Call mine_patterns() first."
            )

        return self.patterns
