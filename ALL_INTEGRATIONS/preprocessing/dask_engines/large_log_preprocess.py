from typing import Optional, List, Literal
import dask.dataframe as dd
from pathlib import Path
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


# ======================================================
# Custom Exceptions (Django / API safe)
# ======================================================

class LargeLogLoadError(RuntimeError):
    """
    Raised when loading a large log file fails
    due to IO, schema, or Dask-related issues.
    """
    pass


class LargeLogFormatError(ValueError):
    """
    Raised when log schema or format is invalid.
    """
    pass


class LargeLogProcessingError(RuntimeError):
    """
    Raised when a distributed preprocessing step fails.
    """
    pass


# ======================================================
# LargeLogPreprocessor
# ======================================================

class LargeLogPreprocessor:
    """
    Scalable preprocessor for very large event logs using Dask.

    This class is the **distributed counterpart** of `LogPreprocessor`
    and follows the exact same API philosophy.

    Design Principles
    -----------------
    • No event-level sampling
    • Fluent / Chainable API
    • Django-safe (Exception driven)
    • Jupyter-friendly (debug mode)
    • Parquet-first, CSV-supported

    Typical Usage
    -------------
    >>> pre = LargeLogPreprocessor(debug=True)
    >>> ddf = (
    ...     pre
    ...         .load("event_log.parquet")
    ...         .auto_clean()
    ...         .sort_by_time()
    ...         .get_ddf()
    ... )
    """

    REQUIRED_COLUMNS = {
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    }

    # --------------------------------------------------
    def __init__(
        self,
        debug: bool = False
    ) -> None:
        """
        Initialize the preprocessor.

        Parameters
        ----------
        debug : bool, default=False
            Enable lightweight console logging (Jupyter/dev only).
        """
        self.debug = debug
        self.path: Optional[str] = None
        self.format: Optional[str] = None
        self.ddf: Optional[dd.DataFrame] = None

    # --------------------------------------------------
    # Internal Logger
    # --------------------------------------------------
    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[LargeLogPreprocessor] {message}")

    # --------------------------------------------------
    # Load
    # --------------------------------------------------
    def load(self, path: str) -> "LargeLogPreprocessor":
        """
        Load event log into a Dask DataFrame.

        Supported formats:
        - parquet
        - csv
        - xes / xes.gz (auto-converted to parquet)
        """

        base_dir = Path(__file__).resolve().parents[3]
        path = base_dir / "data" / path

        if not path.exists():
            raise LargeLogProcessingError(f"File not found: {path}")

        suffix = "".join(path.suffixes).lower()

        # ======================================================
        # Native Dask formats
        # ======================================================
        if suffix == ".parquet":
            self.ddf = dd.read_parquet(path)

        elif suffix == ".csv":
            self.ddf = dd.read_csv(path, assume_missing=True)

        # ======================================================
        # XES / XES.GZ → Pandas → Parquet → Dask
        # ======================================================
        elif suffix in (".xes", ".xes.gz"):

            if self.debug:
                print("[INFO] Loading XES → converting to Parquet")

            # 1️⃣ XES → Pandas
            log = xes_importer.apply(str(path))
            pdf = log_converter.apply(
                log,
                variant=log_converter.Variants.TO_DATA_FRAME
            )

            if pdf.empty:
                raise LargeLogProcessingError("Parsed XES is empty.")

            # 2️⃣ Write temporary parquet
            tmp_dir = path.parent / "__tmp_parquet__"
            tmp_dir.mkdir(exist_ok=True)

            parquet_path = tmp_dir / f"{path.stem}.parquet"
            pdf.to_parquet(parquet_path, index=False)

            # 3️⃣ Parquet → Dask  ✅✅✅ (مهم)
            self.ddf = dd.read_parquet(parquet_path)

        else:
            raise LargeLogFormatError(f"Unsupported format: {suffix}")

        # ✅ sanity check
        if self.ddf is None:
            raise LargeLogProcessingError("Dask DataFrame was not initialized.")

        if self.debug:
            print(f"[INFO] Dask DataFrame loaded with {self.ddf.npartitions} partitions")

        return self



    # --------------------------------------------------
    # Schema validation
    # --------------------------------------------------
    def validate_schema(self) -> "LargeLogPreprocessor":
        """
        Validate presence of core XES columns.
        """
        try:
            missing = self.REQUIRED_COLUMNS - set(self.ddf.columns)
            if missing:
                raise LargeLogFormatError(
                    f"Missing required columns: {sorted(missing)}"
                )

            self._log("Schema validation passed.")
            return self

        except Exception as e:
            raise LargeLogFormatError(str(e)) from None

    # --------------------------------------------------
    # Drop columns
    # --------------------------------------------------
    def drop_columns(self, columns: List[str]) -> "LargeLogPreprocessor":
        """
        Drop unnecessary columns safely.
        """
        try:
            self.ddf = self.ddf.drop(columns=columns, errors="ignore")
            self._log(f"Dropped columns: {columns}")
            return self

        except Exception as e:
            raise LargeLogProcessingError(
                f"Failed to drop columns {columns}: {e}"
            ) from None

    # --------------------------------------------------
    # Drop constant columns (safe heuristic)
    # --------------------------------------------------
    def drop_constant_columns(
        self,
        sample_cases: int = 1000
    ) -> "LargeLogPreprocessor":
        """
        Remove columns that are constant across sampled CASES
        (CASE-level safe, process-aware).
        """
        try:
            case_col = "case:concept:name"

            sample_df = (
                self.ddf
                    .drop_duplicates(subset=[case_col])
                    .head(sample_cases)
                    .compute()
            )

            constant_cols = [
                c for c in sample_df.columns
                if sample_df[c].nunique(dropna=False) <= 1
            ]

            self.ddf = self.ddf.drop(columns=constant_cols, errors="ignore")
            self._log(f"Constant columns removed: {constant_cols}")
            return self

        except Exception as e:
            raise LargeLogProcessingError(
                f"Failed to drop constant columns: {e}"
            ) from None

    # --------------------------------------------------
    # auto_clean (distributed-safe)
    # --------------------------------------------------
    def auto_clean(self) -> "LargeLogPreprocessor":
        """
        Minimal, distributed-safe cleaning.

        Operations
        ----------
        • Drop events with missing critical fields
        • Normalize activity names
        """
        try:
            self.validate_schema()

            self.ddf = self.ddf.dropna(subset=list(self.REQUIRED_COLUMNS))

            self.ddf["concept:name"] = (
                self.ddf["concept:name"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
            )

            self._log("auto_clean executed.")
            return self

        except Exception as e:
            raise LargeLogProcessingError(
                f"auto_clean failed: {e}"
            ) from None

    # --------------------------------------------------
    # Sort events (IEEE trace semantics)
    # --------------------------------------------------
    def sort_by_time(self) -> "LargeLogPreprocessor":
        """
        Sort events by Case ID and timestamp.
        """
        try:
            self.ddf = self.ddf.map_partitions(
                lambda df: df.sort_values(
                    ["case:concept:name", "time:timestamp"]
                )
            )

            self._log("Events sorted by case and time.")
            return self

        except Exception as e:
            raise LargeLogProcessingError(
                f"Failed to sort events: {e}"
            ) from None

    # --------------------------------------------------
    # Export helpers
    # --------------------------------------------------
    def get_ddf(self):
        """
        Return the active Dask DataFrame.
        """
        if self.ddf is None:
            raise LargeLogProcessingError("No Dask DataFrame available.")
        return self.ddf

    def persist(self, out_path: str) -> None:
        """
        Persist cleaned log as Parquet dataset.
        """
        try:
            self.ddf.to_parquet(
                out_path,
                write_index=False,
                overwrite=True
            )
            self._log(f"Clean log persisted to {out_path}")

        except Exception as e:
            raise LargeLogProcessingError(
                f"Failed to persist cleaned log: {e}"
            ) from None
