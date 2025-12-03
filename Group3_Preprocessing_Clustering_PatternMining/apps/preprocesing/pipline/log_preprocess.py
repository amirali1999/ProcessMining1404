import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from rapidfuzz import fuzz, process
from typing import Literal, List, Dict, Union, Any, Optional
from pathlib import Path

from requests.utils import select_proxy
from tabulate import tabulate

class LogPreprocessor:
    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        self.file_path = None
        self._suggestions = None
        self.log = None
        self.df_original = df
        self.df = df
        self.selected_columns = [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "org:resource",
            "lifecycle:transition",
        ]
        self._log_history = None
        self.debug = False

    def _log_info(
            self,
            message: str,
            level: Literal["INFO", "WARN", "ERROR", "STEP", "DEBUG"] = "INFO",
            show_time: bool = True,
            color: bool = True
    ) -> None:
        """
        Internal lightweight logger for diagnostic and progress messages.

        Parameters
        ----------
        message : str
            The message text to display.
        level : {'INFO', 'WARN', 'ERROR', 'STEP', 'DEBUG'}, default='INFO'
            Log severity or context label.
        show_time : bool, default=True
            Whether to prepend a time stamp (HH:MM:SS).
        color : bool, default=True
            Whether to display colored output (ANSI ANSI-compatible terminals only).

        Notes
        -----
        • Controlled by `self.debug` flag: messages are printed only if debug=True
        • Used internally by almost all preprocessing functions
        • Avoids overhead of Python's built-in logging module for performance-critical data ops.
        """
        try:
            return
            # Respect the debug toggle
            if not getattr(self, "debug", False):
                return

            from datetime import datetime

            # Prepare timestamp and label
            ts = datetime.now().strftime("%H:%M:%S") if show_time else ""
            label = f"[{level}]"
            prefix = f"{ts} {label}".strip()

            # --- ANSI color codes ---
            color_map = {
                "INFO": "\033[92m",  # green
                "WARN": "\033[93m",  # yellow
                "ERROR": "\033[91m",  # red
                "STEP": "\033[94m",  # blue
                "DEBUG": "\033[95m",  # magenta
                "RESET": "\033[0m"
            }

            # Apply color if requested
            if color and level in color_map:
                prefix = f"{color_map[level]}{prefix}{color_map['RESET']}"

            print(f"{prefix} {message}")

            # Optionally, log to internal history for debugging sessions
            if not hasattr(self, "_log_history"):
                self._log_history = []
            self._log_history.append(
                {"timestamp": ts, "level": level, "message": message}
            )

        except Exception as e:
            # Fail silently - logger must never break the main pipeline
            if getattr(self, "debug", False):
                print(f"[LoggingError] Failed in _log_info(): {e}")

    def load(
        self,
        filename: str,
        mode: Literal["strict", "tolerant"] = "strict",
        inplace: bool = True
    ) -> pd.DataFrame:
        """
        Load an event log from the ./data/ directory and convert to a PM4Py EventLog.

        Parameters
        ----------
        filename : str
            Target file name (supports .csv, .xes, .xes.gz).
        mode : {"strict", "tolerant"}, default="strict"
            Error handling behavior (raise or debug-print).
        inplace : bool, default=True
            If True, assigns results to instance attributes.

        Returns
        -------
        pd.DataFrame
            Loaded event log in PM4Py DataFrame format.

        Raises
        ------
        LogFileNotFoundError
            If the file path does not exist.
        LogFormatError
            If the file format is unsupported.
        LogLoadError
            If parsing or conversion fails.
        """

        base_dir = Path(__file__).resolve().parents[3]
        self.file_path = base_dir / "data" / filename

        if not self.file_path.exists():
            if mode == "tolerant" and getattr(self, "debug", False):
                print(f"[WARN] File not found: {self.file_path}")
                return pd.DataFrame()
            raise LogFileNotFoundError(f"File not found: {self.file_path}") from None

        ext = "".join(self.file_path.suffixes).lower()

        try:
            if ext in (".xes", ".xes.gz"):
                log = xes_importer.apply(str(self.file_path))
                df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

            elif ext == ".csv":
                df = pd.read_csv(self.file_path)
                df = dataframe_utils.convert_timestamp_columns_in_df(df)
                log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
                df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

            else:
                raise LogFormatError(f"Unsupported file format: {ext}")

            if inplace:
                self.df_original = df.copy()
                self.df = df
                self.log = log

            return df

        except (LogFileNotFoundError, LogFormatError):
            raise

        except Exception as e:
            if mode == "tolerant" and getattr(self, "debug", False):
                print(f"[ERROR] Failed to load {self.file_path}: {e}")
                return pd.DataFrame()
            raise LogLoadError(f"Failed to load log file: {self.file_path}") from None

    def reset_to_original(
            self,
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Restore the working DataFrame to its original loaded state.

        This method resets the active DataFrame (`self.df`) so that it becomes
        identical to the originally loaded version stored in `self.df_original`.
        Useful when multiple preprocessing steps have been performed and the user
        wants to revert to the raw dataset without reloading from disk.

        Parameters
        ----------
        inplace : bool, default=False
            If True, permanently replaces `self.df` with a copy of
            `self.df_original`.
            If False, returns a copy of the original DataFrame
            while leaving `self.df` unchanged.

        Returns
        -------
        pd.DataFrame
            The restored DataFrame matching the initial loaded state.

        Raises
        ------
        LogLoadError
            If the original DataFrame was never initialized or is not available.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df_original") or self.df_original is None:
                raise LogLoadError("Original DataFrame not found. Ensure load() was called first.")

            # --- Core restoration logic ---
            restored_df = self.df_original.copy()

            # --- Apply inplace behavior ---
            if inplace:
                self.df = restored_df

            # --- Optional debug info ---
            if getattr(self, "debug", False):
                scope = "inplace" if inplace else "returned as copy"
                print(f"[INFO] reset_to_original executed ({scope}).")

            return restored_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] reset_to_original() failed: {e}")
            raise LogLoadError("Failed to restore DataFrame to original state.") from None

    def list_columns(self) -> List[str]:
        """
        Display all column names of the current dataframe as a list.

        Returns
        -------
        list of str
            List of column names from the dataframe.

        Raises
        ------
        LogLoadError
            If dataframe is not initialized or invalid.
        """
        try:
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before list_columns().")

            cols = list(self.df.columns)
            print(", ".join(cols))  # e.g. name, concept, time, ...
            return cols

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] list_columns() failed: {e}")
            raise LogLoadError("Failed to list dataframe columns.") from None

    def list_columns_types(
            self,
            show_types: bool = True,
            output_format: Literal["dataframe", "dict", "print"] = "print"
    ) -> pd.DataFrame | dict | None:
        """
        List all DataFrame columns with detected data types.

        Parameters
        ----------
        show_types : bool, default=True
            Whether to display detected type labels (numeric, text, datetime, category).
        output_format : {'dataframe', 'dict', 'print'}, default='dataframe'
            Determines the format of the output:
            - 'dataframe': returns a pandas DataFrame
            - 'dict': returns a Python dictionary
            - 'print': prints the formatted result table in console/Jupyter

        Returns
        -------
        pd.DataFrame or dict or None
            Structured output of columns and their inferred types.

        Raises
        ------
        LogLoadError
            If the DataFrame attribute is missing or inaccessible.
        """

        try:
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("No DataFrame found in LogPreprocessor instance.")

            df = self.df.copy()
            col_info = []

            for col in df.columns:
                dtype = df[col].dtype
                if show_types:
                    if pd.api.types.is_numeric_dtype(dtype):
                        kind = "numeric"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        kind = "datetime"
                    elif pd.api.types.is_categorical_dtype(dtype):
                        kind = "category"
                    else:
                        kind = "text"
                else:
                    kind = None

                col_info.append({"column": col, "dtype": str(dtype), "detected_type": kind})

            result_df = pd.DataFrame(col_info)

            if output_format == "dataframe":
                return result_df
            elif output_format == "dict":
                return result_df.to_dict(orient="records")
            elif output_format == "print":
                print(tabulate(result_df, headers="keys", tablefmt="psql"))
            return None

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] list_columns() failed: {e}")
            raise LogLoadError("Failed to list columns and their types.") from None

    def drop_columns(
            self,
            columns: Optional[str | list[str]] = None,
            inplace: bool = True,
            keep_mode: bool = False
    ) -> pd.DataFrame:
        """
        Remove or selectively keep columns from the active DataFrame.

        When called without parameters, keeps only standard XES event‑log columns.

        Parameters
        ----------
        columns : str or list of str, optional
            Column(s) to process. Behavior depends on `keep_mode`:
            - If `keep_mode=False`: these columns will be **removed**.
            - If `keep_mode=True`: these columns will be **kept**, others dropped.
            - If `columns=None`: retains only standard XES columns
              ['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource'].
        inplace : bool, default=True
            Whether to modify the current DataFrame directly.
            The resulting DataFrame is always returned for visibility.
        keep_mode : bool, default=False
            If True, only provided columns are kept; others are dropped.

        Returns
        -------
        pd.DataFrame
            Resulting DataFrame after column removal or selective retention.

        Raises
        ------
        LogLoadError
            If DataFrame is not initialized or operation fails.
        """
        try:
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before drop_columns().")

            df = self.df.copy()
            existing_cols = df.columns.tolist()

            # --- Mode 3: Automatic XES standard mode ---
            if columns is None:
                standard_xes_cols = [
                    "case:concept:name",
                    "concept:name",
                    "time:timestamp",
                    "org:resource",
                    "lifecycle:transition",
                ]
                valid_keep = [c for c in standard_xes_cols if c in existing_cols]
                cleaned_df = df[valid_keep]
                action = f"Auto‑kept XES standard columns: {valid_keep}"

            else:
                cols = [columns] if isinstance(columns, str) else list(columns)

                if keep_mode:
                    # --- Mode 2: Keep only these columns ---
                    valid_cols = [c for c in cols if c in existing_cols]
                    cleaned_df = df[valid_cols]
                    action = f"Kept only columns: {valid_cols}"
                else:
                    # --- Mode 1: Drop these columns ---
                    cleaned_df = df.drop(columns=cols, errors="ignore")
                    action = f"Dropped columns: {cols}"

            if inplace:
                self.df = cleaned_df

            if getattr(self, "debug", False):
                print(f"[INFO] drop_columns executed → {action}")
                print(cleaned_df.head())

            return cleaned_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] drop_columns() failed (keep_mode={keep_mode}) for columns={columns}: {e}")
            raise LogLoadError("Failed to drop/keep columns.") from None

    def drop_constant_columns(
            self,
            inplace: bool = False,
            null_threshold: float = 0.0
    ) -> Any:
        """
        Remove columns with constant values OR columns whose null-ratio
        exceeds a given threshold.

        Parameters
        ----------
        inplace : bool, default False
            If True, modifies the dataframe in place.
            If False, returns a cleaned copy.
        null_threshold : float, default 0.0
            Maximum allowed percentage (0–1) of null values.
            Columns exceeding this ratio will be dropped.

        Returns
        -------
        pd.DataFrame
            DataFrame without constant columns and high-null columns.

        Raises
        ------
        LogLoadError
            If dataframe is not initialized or operation fails.
        """
        try:
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before drop_constant_columns().")

            df = self.df.copy()

            # --- 1) Detect constant columns ---
            constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

            # --- 2) Detect high-null columns ---
            high_null_cols = []
            if null_threshold > 0:
                row_count = len(df)
                for col in df.columns:
                    null_ratio = df[col].isna().sum() / row_count
                    if null_ratio > null_threshold:
                        high_null_cols.append(col)

            # combine both sets
            cols_to_drop = list(set(constant_cols + high_null_cols))

            cleaned_df = df.drop(columns=cols_to_drop, errors="ignore")

            # Debug output
            if getattr(self, "debug", False):
                print(f"[INFO] Constant columns dropped: {constant_cols}")
                print(f"[INFO] High-null columns dropped (>{null_threshold * 100}% null): {high_null_cols}")
                print(cleaned_df.head())

            # Inplace behavior
            if inplace:
                self.df = cleaned_df
                return cleaned_df

            return cleaned_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] drop_constant_columns() failed: {e}")
            raise LogLoadError("Failed to drop constant/high-null columns.") from None

    def unique_summary(
            self,
            columns: Union[str, List[str]],
            top_n: int = 20,
            plot: bool = True,
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Generate a frequency summary of unique values or column combinations.

        Parameters
        ----------
        columns : str or list of str
            Target column(s) to compute unique value frequencies. Multiple columns
            are treated as composite keys.
        top_n : int, default=20
            Number of top frequent values to include in the plot.
        plot : bool, default=True
            Whether to visualize the frequency distribution using matplotlib.
        inplace : bool, default=False
            (Kept for API consistency) Does not modify self.df. Always returns DataFrame.

        Returns
        -------
        pd.DataFrame
            A frequency summary with two columns: ['unique_key', 'count'].
            Always visible in Jupyter and reusable as Matplotlib input.

        Raises
        ------
        LogLoadError
            If dataframe is not initialized or one of the columns does not exist.
        """
        try:
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before unique_summary().")

            # Normalize column list
            cols = [columns] if isinstance(columns, str) else columns
            for col in cols:
                if col not in self.df.columns:
                    raise LogLoadError(f"Column '{col}' not found in DataFrame.")

            # Create composite key if multiple columns passed
            composite = self.df[cols].astype(str).agg("_".join, axis=1)
            summary_df = composite.value_counts().rename_axis("unique_key").reset_index(name="count")

            # Display result in Jupyter automatically (returning ensures table rendering)
            if plot:
                plt.figure(figsize=(10, 5))
                subset = summary_df.head(top_n)
                plt.bar(subset["unique_key"], subset["count"], color="#4682B4")
                plt.title(f"Top-{top_n} Unique Values for {cols}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()

            if getattr(self, "debug", False):
                print(f"[INFO] unique_summary computed for {cols}")
                print(summary_df.head())

            # Always return the summary DataFrame for further matplotlib usage
            return summary_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] unique_summary() failed for columns={columns}: {e}")
            raise LogLoadError(f"Failed to compute unique summary for {columns}") from None

    def rename_column(
            self,
            old_name: str,
            new_name: str,
            inplace: bool = True
    ) -> pd.DataFrame:
        """
        Rename a single column in the active DataFrame.

        Parameters
        ----------
        old_name : str
            Existing column name that should be replaced.
        new_name : str
            Target name to assign to the column.
        inplace : bool, default=True
            Whether to modify the working dataframe in place.
            The modified or generated DataFrame is always returned
            for visibility (Jupyter/Django-safe behavior).

        Returns
        -------
        pd.DataFrame
            DataFrame in which the column has been renamed,
            regardless of the inplace flag status.

        Raises
        ------
        LogLoadError
            If DataFrame is not initialized or old_name does not exist.
        """
        try:
            # Validate dataframe existence
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before rename_column().")

            # Validate target column existence
            if old_name not in self.df.columns:
                raise LogLoadError(f"Column '{old_name}' not found in DataFrame.")

            # Perform renaming
            renamed_df = self.df.rename(columns={old_name: new_name})

            # If inplace modification requested
            if inplace:
                self.df = renamed_df

            if getattr(self, "debug", False):
                print(f"[INFO] Renamed column '{old_name}' → '{new_name}'")

            # Always return DataFrame for visibility
            return renamed_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] rename_column() failed: {e}")
            raise LogLoadError(f"Failed to rename column '{old_name}' to '{new_name}'.") from None

    def remove_duplicates(
            self,
            columns: Optional[Union[str, List[str]]] = None,
            groupby: Optional[Union[str, List[str]]] = None,
            keep: str = "first",
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from the active DataFrame.

        Optionally supports group-wise duplicate removal for contextual
        scenarios such as process instance segmentation.

        Parameters
        ----------
        columns : str or list of str, optional
            Columns used to identify duplicates.
            If None, all columns are considered.
        keep : {"first", "last", False}, default="first"
            Determines which duplicates to retain.
        groupby : str or list of str, optional
            Optional column(s) to group by before duplicate detection.
            Each group is processed independently.
        inplace : bool, default=False
            If True, modifies `self.df` in place. Otherwise returns a
            new DataFrame copy (recommended default for shared services).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with duplicates removed, always returned
            for visibility in both Django and Jupyter contexts.

        Raises
        ------
        LogLoadError
            If DataFrame is not initialized or duplicate removal fails.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before remove_duplicates().")

            df = self.df.copy()

            # --- Normalize columns argument ---
            if isinstance(columns, str):
                columns = [columns]

            # --- Non-grouped duplicate removal ---
            if groupby is None:
                before = len(df)
                cleaned_df = df.drop_duplicates(subset=columns, keep=keep)
                after = len(cleaned_df)
                removed = before - after

                if getattr(self, "debug", False):
                    print(f"[INFO] Removed {removed} global duplicates "
                          f"based on {columns if columns else 'all columns'}.")
            else:
                # --- Group-wise duplicate removal ---
                if isinstance(groupby, str):
                    groupby = [groupby]

                before = len(df)
                cleaned_df = (
                    df.groupby(groupby, group_keys=False)
                    .apply(lambda g: g.drop_duplicates(subset=columns, keep=keep))
                    .reset_index(drop=True)[df.columns]  # Future‑proof: keep group columns
                )
                after = len(cleaned_df)
                removed = before - after

                if getattr(self, "debug", False):
                    print(f"[INFO] Removed {removed} duplicates within groups "
                          f"{groupby} based on {columns if columns else 'all columns'}.")

            # --- Apply inplace logic ---
            if inplace:
                self.df = cleaned_df

            return cleaned_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] remove_duplicates() failed: {e}")
            raise LogLoadError("Failed to remove duplicates.") from None

    def remove_by_condition(
            self,
            condition: Optional[str] = None,
            column: Optional[str] = None,
            allowed_values: Optional[list] = None,
            mode: Literal["in", "not_in"] = "not_in",
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Remove rows matching a condition string or based on list membership.

        Parameters
        ----------
        condition : str, optional
            Pandas query syntax, e.g. "`org:group` in ['A','B']" or "`org:group` not in ['A','B']".
        column : str, optional
            Column name to check values against allowed_values.
        allowed_values : list, optional
            List of values for inclusion/exclusion.
        mode : {'in', 'not_in'}, default='not_in'
            Used only if column+allowed_values specified.
        inplace : bool, default=False
            Whether to modify the DataFrame in-place.
        """
        if getattr(self, "df", None) is None:
            raise LogLoadError("DataFrame not initialized.")

        df = self.df.copy()

        try:
            if condition:
                matched = df.query(condition)
                # remove matched rows
                df = df.loc[~df.index.isin(matched.index)]
            elif column and allowed_values is not None:
                if mode == "in":
                    df = df.loc[~df[column].isin(allowed_values)]  # remove rows in list
                elif mode == "not_in":
                    df = df.loc[df[column].isin(allowed_values)]  # keep only in list
                else:
                    raise LogFormatError(f"Unknown mode '{mode}'.")
            else:
                raise LogFormatError("Either condition or column+allowed_values required.")

            if inplace:
                self.df = df
            return df

        except Exception as e:
            raise LogLoadError(f"Failed to apply remove_by_condition: {e}") from None

    def anonymize_column(
            self,
            column: str,
            mode: Literal["numeric", "alphabetic"] = "numeric",
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Anonymize the values of a specified column by replacing them with unique identifiers.

        Parameters
        ----------
        column : str
            The column name in the DataFrame to be anonymized.
        mode : {'numeric', 'alphabetic'}, default='numeric'
            Encoding style:
            - 'numeric': 1, 2, 3, ...
            - 'alphabetic': A, B, C, ..., Z, AA, AB, ...
        inplace : bool, default=False
            If False, returns a modified copy of DataFrame without changing self.df.

        Returns
        -------
        pd.DataFrame
            DataFrame with the anonymized column.

        Raises
        ------
        LogLoadError
            If the specified column does not exist or conversion fails.
        """

        try:
            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame found in this instance.")
            if column not in self.df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            df = self.df.copy()

            # extract unique values
            uniques = df[column].dropna().unique()
            id_map = {}

            if mode == "numeric":
                id_map = {val: i + 1 for i, val in enumerate(uniques)}
            elif mode == "alphabetic":
                # recursive alphabetic labels (A, B, ..., Z, AA, AB, ...)
                def num_to_letters(n: int) -> str:
                    letters = []
                    while n > 0:
                        n, remainder = divmod(n - 1, 26)
                        letters.append(chr(65 + remainder))
                    return "".join(reversed(letters))

                id_map = {val: num_to_letters(i + 1) for i, val in enumerate(uniques)}
            else:
                raise LogLoadError("Mode must be either 'numeric' or 'alphabetic'.")

            # apply mapping safely
            df[column] = df[column].map(id_map)

            if inplace:
                self.df = df
                return self.df
            else:
                return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] anonymize_column() failed: {e}")
            raise LogLoadError(f"Failed to anonymize column '{column}'.") from None

    def convert_column_type(
            self,
            column: str,
            target_type: Literal["str", "numeric", "datetime", "bool"],
            on_fail: Literal["nan", "delete", "default", "medium", "max", "min", "nothing"] = "nan",
            default_value: Any = None,
            inplace: bool = True,
            datetime_formats: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Convert the data type of a single column with controlled fallback behavior.

        Parameters
        ----------
        column : str
            Column name to convert.
        target_type : {'str', 'numeric', 'datetime', 'bool'}
            Desired target data type.
        on_fail : {'nan', 'delete', 'default', 'medium', 'max', 'min', 'nothing'}, default='nan'
            Fallback action applied when conversion cannot be performed.
        default_value : Any, optional
            Substitution value used only when on_fail='default'.
        inplace : bool, default=True
            Whether to modify the dataframe in-place.
            Regardless of this flag, the resulting DataFrame is always returned.
        datetime_formats : list of str, optional
            Custom parsing formats for datetime conversion.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame visible in both Jupyter and Django contexts.

        Raises
        ------
        LogLoadError
            If the dataframe or column is missing.
        LogFormatError
            If an incompatible fallback mode (e.g., medium/min/max on non-numeric types) is chosen.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before convert_column_type().")
            if column not in self.df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            df = self.df.copy()
            series = df[column]

            if datetime_formats is None:
                datetime_formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]

            converted_series = series.copy()

            # --- Type Conversion ---
            try:
                if target_type == "str":
                    converted_series = series.astype(str)
                elif target_type == "numeric":
                    converted_series = pd.to_numeric(series, errors="coerce")
                elif target_type == "datetime":
                    success = False
                    for fmt in datetime_formats:
                        try:
                            converted_series = pd.to_datetime(series, format=fmt, errors="coerce")
                            success = True
                            break
                        except Exception:
                            continue
                    if not success:
                        converted_series = pd.to_datetime(series, errors="coerce")
                elif target_type == "bool":
                    # Allow basic normalization of truthy/falsy values
                    converted_series = series.map(lambda x: True if str(x).strip().lower() in ["true", "1", "yes"]
                    else False if str(x).strip().lower() in ["false", "0", "no"]
                    else np.nan)
                else:
                    raise LogFormatError(f"Unsupported target_type '{target_type}'.")
            except Exception as e:
                if getattr(self, "debug", False):
                    print(f"[ERROR] Conversion failed for column '{column}': {e}")

            # --- Handle failed conversions ---
            failed_mask = converted_series.isna()
            if failed_mask.any():
                if on_fail == "nan":
                    df[column] = converted_series
                elif on_fail == "delete":
                    df = df.loc[~failed_mask]
                elif on_fail == "default":
                    df.loc[failed_mask, column] = default_value
                elif on_fail in ["medium", "max", "min"]:
                    # Validate numeric compatibility
                    if pd.api.types.is_numeric_dtype(converted_series):
                        if on_fail == "medium":
                            fill_value = converted_series.mean()
                        elif on_fail == "max":
                            fill_value = converted_series.max()
                        elif on_fail == "min":
                            fill_value = converted_series.min()
                        df.loc[failed_mask, column] = fill_value
                    else:
                        raise LogFormatError(
                            f"Incompatible fallback '{on_fail}' for non-numeric column '{column}'."
                        )
                elif on_fail == "nothing":
                    pass
                else:
                    raise LogFormatError(f"Unknown fallback mode: '{on_fail}'")

            if inplace:
                self.df = df

            if getattr(self, "debug", False):
                print(f"[INFO] Conversion applied to '{column}' → {target_type} (on_fail={on_fail})")

            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] convert_column_type() failed: {e}")
            raise LogLoadError(f"Failed to convert column '{column}' to {target_type}.") from None

    def replace_values(
            self,
            columns: Union[str, List[str]],
            to_replace: Optional[Union[Any, List[Any]]] = None,
            conditions: Optional[Union[str, List[Union[str, tuple]]]] = None,
            new_value: Any = None,
            inplace: bool = True
    ) -> pd.DataFrame:
        """
        Replace specific values or apply conditional replacements across DataFrame columns.

        Extends basic value replacement with conditional logic support. The user can define
        one or multiple logical expressions referencing other columns (similar to a WHERE clause).

        Parameters
        ----------
        columns : str or list of str
            Target column(s) to apply replacements. Can be a single column or list.
        to_replace : object or list, optional
            Value(s) to be replaced (ignored if 'conditions' is provided).
        new_value : object, optional
            Replacement value to assign for matching cases.
        conditions : str | list[str | tuple], optional
            Conditional expressions used for row selection. String expressions may follow
            Pandas query syntax (e.g. `"A=='X' and B>3"`). Alternatively, a list of tuples like
            `[("A","==","X"), ("B",">",3)]` is accepted and auto‑combined with logical AND.
        inplace : bool, default=True
            If True, apply modifications on self.df. Otherwise, return a modified copy.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with replacement results visible in both Django and Jupyter contexts.

        Raises
        ------
        LogLoadError
            If the DataFrame or columns are uninitialized / missing.
        LogFormatError
            For malformed conditions or incompatible replacement operations.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before replace_values().")

            df = self.df.copy()

            if isinstance(columns, str):
                columns = [columns]

            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise LogLoadError(f"Column(s) {missing_cols} not found in DataFrame.")

            # --- Build condition mask if provided ---
            if conditions is not None:
                try:
                    if isinstance(conditions, str):
                        mask = df.query(conditions).index
                    else:
                        expr_parts = []
                        for cond in conditions:
                            if isinstance(cond, tuple) and len(cond) == 3:
                                col, op, val = cond
                                val_str = f"'{val}'" if isinstance(val, str) else str(val)
                                expr_parts.append(f"`{col}` {op} {val_str}")
                            elif isinstance(cond, str):
                                expr_parts.append(cond)
                            else:
                                raise LogFormatError(f"Invalid condition format: {cond}")
                        full_expr = " and ".join(expr_parts)
                        mask = df.query(full_expr).index
                except Exception as e:
                    raise LogFormatError(f"Failed to evaluate condition(s): {e}") from None

                # --- Apply conditional replacement ---
                for col in columns:
                    df.loc[mask, col] = new_value

                if getattr(self, "debug", False):
                    print(f"[INFO] Conditional replacement applied on {columns} for {len(mask)} matching rows.")

            else:
                # --- Simple value replacement (legacy behavior) ---
                for col in columns:
                    try:
                        df[col] = df[col].replace(to_replace, new_value)
                    except Exception as e:
                        raise LogFormatError(f"Failed to replace values in column '{col}': {e}") from None

                    if getattr(self, "debug", False):
                        print(f"[INFO] Direct replacement applied on '{col}' for {to_replace} → {new_value}")

            # --- Inplace handling ---
            if inplace:
                self.df = df

            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] replace_values() failed: {e}")
            raise LogLoadError("Failed to apply replace_values operation.") from None

    def handle_null_values(
            self,
            column: str,
            on_null: Literal["delete", "default", "medium", "max", "min"] = "delete",
            default_value: Any = None,
            inplace: bool = True
    ) -> pd.DataFrame:
        """
        Handle rows with null/NaN values in a single column according to the given strategy.

        Parameters
        ----------
        column : str
            Target column name to process for null/NaN handling.
        on_null : {'delete', 'default', 'medium', 'max', 'min'}, default='delete'
            Strategy to apply when encountering null values:
            - 'delete' : Remove rows containing nulls in this column.
            - 'default': Replace nulls with a provided default value.
            - 'medium' : Replace nulls with the column mean (numeric only).
            - 'max'    : Replace nulls with the column maximum (numeric only).
            - 'min'    : Replace nulls with the column minimum (numeric only).
        default_value : Any, optional
            Value used only when on_null='default'.
        inplace : bool, default=True
            Whether to apply operation directly to self.df.
            Regardless, the resulting DataFrame is always returned.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame reflecting applied null-handling strategy.

        Raises
        ------
        LogLoadError
            If the DataFrame is not initialized or column missing.
        LogFormatError
            If a numeric-only operation is attempted on non-numeric column type.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before handle_null_values().")

            df = self.df.copy()

            if column not in df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            # Identify null rows
            null_mask = df[column].isna()

            if not null_mask.any():
                if getattr(self, "debug", False):
                    print(f"[INFO] No null values found in '{column}'. Nothing to process.")
                return df

            # --- Determine Column Type for compatibility check ---
            col_dtype = df[column].dtype

            # --- Apply null-handling strategy ---
            if on_null == "delete":
                df = df.loc[~null_mask]

            elif on_null == "default":
                df.loc[null_mask, column] = default_value

            elif on_null in ["medium", "max", "min"]:
                # Numeric operations only
                if not pd.api.types.is_numeric_dtype(col_dtype):
                    raise LogFormatError(
                        f"Incompatible on_null='{on_null}' for non-numeric column '{column}'."
                    )
                if on_null == "medium":
                    fill_value = df[column].mean()
                elif on_null == "max":
                    fill_value = df[column].max()
                elif on_null == "min":
                    fill_value = df[column].min()

                df.loc[null_mask, column] = fill_value

            else:
                raise LogFormatError(f"Unsupported on_null mode '{on_null}' provided.")

            # --- Update working copy and return ---
            if inplace:
                self.df = df

            if getattr(self, "debug", False):
                print(f"[INFO] handle_null_values applied on '{column}' (strategy={on_null})")

            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] handle_null_values() failed: {e}")
            raise LogLoadError(f"Failed during null value handling for '{column}'.") from None

    def filter_by_range_and_allowed(
            self,
            column: str,
            allowed_values: Optional[List[Any]] = None,
            range_min: Optional[Any] = None,
            range_max: Optional[Any] = None,
            on_outside: Literal["delete", "nan", "default", "medium", "max", "min", "show"] = "nan",
            default_value: Any = None,
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter or manage rows of a column based on a set of allowed values and a numeric/datetime range.

        Parameters
        ----------
        column : str
            Target column name.
        allowed_values : list, optional
            List of allowed categorical values. If None, only range filtering is applied.
        range_min : Any, optional
            Lower bound of valid numeric or datetime range.
        range_max : Any, optional
            Upper bound of valid numeric or datetime range.
        on_outside : {'delete', 'nan', 'default', 'medium', 'max', 'min', 'show'}, default='nan'
            Action to apply when a value is outside allowed range/set:
            - 'delete': Remove those records entirely.
            - 'nan': Set offending field to NaN (default).
            - 'default': Replace with a provided default value.
            - 'medium': Replace with column mean (numeric only).
            - 'max': Replace with column maximum (numeric only).
            - 'min': Replace with column minimum (numeric only).
            - 'show': Return out-of-range records only.
        default_value : Any, optional
            Value used when `on_outside='default'`.
        inplace : bool, default=False (exception)
            This function does not update `self.df` by default.

        Returns
        -------
        pd.DataFrame
            Filtered or modified DataFrame.

        Raises
        ------
        LogLoadError
            If DataFrame or column is missing.
        LogFormatError
            For incompatible operations on non-numeric columns.
        """
        try:
            # --- Validation ---
            if not hasattr(self, "df") or self.df is None:
                raise LogLoadError("DataFrame not initialized before filter_by_range_and_allowed().")
            if column not in self.df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            df = self.df.copy()
            series = df[column]

            # --- Automatic detection for datetime columns ---
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            # Attempt to convert if column appears date-like but not stored as datetime
            if not is_datetime and series.dtype == object:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        parsed = pd.to_datetime(series, errors="coerce")
                        # if enough values converted successfully (>=70%)
                        if parsed.notna().mean() >= 0.7:
                            series = parsed
                            df[column] = parsed
                            is_datetime = True
                            if getattr(self, "debug", False):
                                print(f"[INFO] Column '{column}' automatically detected as datetime.")
                    except Exception:
                        pass

            # Convert range_min/max appropriately if datetime column
            if is_datetime:
                if range_min is not None:
                    range_min = pd.to_datetime(range_min, errors="coerce")
                if range_max is not None:
                    range_max = pd.to_datetime(range_max, errors="coerce")

            # --- Determine mask for values outside range or allowed set ---
            outside_mask = pd.Series(False, index=df.index)

            if allowed_values is not None:
                outside_mask |= ~series.isin(allowed_values)
            if range_min is not None:
                outside_mask |= series < range_min
            if range_max is not None:
                outside_mask |= series > range_max

            # --- Handle 'show' mode ---
            if on_outside == "show":
                out_df = df.loc[outside_mask]
                if getattr(self, "debug", False):
                    print(f"[INFO] Showing {len(out_df)} records outside valid range/set for '{column}'")
                return out_df

            # --- Early exit if nothing outside ---
            if not outside_mask.any():
                if getattr(self, "debug", False):
                    print(f"[INFO] All values within range/set for '{column}'.")
                return df

            # --- Apply correction logic ---
            if on_outside == "delete":
                df = df.loc[~outside_mask]

            elif on_outside == "nan":
                df.loc[outside_mask, column] = np.nan

            elif on_outside == "default":
                df.loc[outside_mask, column] = default_value

            elif on_outside in ["medium", "max", "min"]:
                # Valid only for numeric columns (not datetime or string)
                if not pd.api.types.is_numeric_dtype(series):
                    raise LogFormatError(
                        f"Incompatible action '{on_outside}' for non-numeric column '{column}'."
                    )
                if on_outside == "medium":
                    fill_val = series.mean()
                elif on_outside == "max":
                    fill_val = series.max()
                elif on_outside == "min":
                    fill_val = series.min()
                df.loc[outside_mask, column] = fill_val

            else:
                raise LogFormatError(f"Unknown on_outside mode '{on_outside}'.")

            if inplace:
                self.df = df

            if getattr(self, "debug", False):
                print(f"[INFO] filter_by_range_and_allowed applied on '{column}' (mode={on_outside})")

            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] filter_by_range_and_allowed() failed: {e}")
            raise LogLoadError(f"Failed during range filtering for '{column}'.") from None

    def handle_outliers(
            self,
            column: str,
            deviation_percent: float,
            groupby_column: Optional[str] = None,
            on_outside: Literal["delete", "nan", "default", "medium", "max", "min", "show"] = "delete",
            default_value: Optional[float] = None,
            inplace: bool = False,
    ) -> pd.DataFrame:
        """
            Detect and handle statistical outliers based on percentage deviation from mean.
            Supports both numeric and datetime data. Group-specific or global deviation can be applied.

            Parameters
            ----------
            column : str
                Column name to analyze.
            deviation_percent : float
                Percentage threshold for deviation from mean.
            groupby_column : str, optional
                Optional column to perform analysis per group.
            on_outside : {'delete', 'nan', 'default', 'medium', 'max', 'min', 'show'}, default='delete'
                Defines action for rows found outside deviation range.
            default_value : float, optional
                Value to insert when `on_outside='default'`.
            inplace : bool, default=False
                Whether to modify self.df in place.

            Returns
            -------
            pd.DataFrame
                Resulting DataFrame after applying outlier handling.

            Raises
            ------
            LogLoadError
                If operation fails due to incompatible column types or other processing error.
            """

        try:
            df = self.df.copy()

            if column not in df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            series = df[column]

            # Determine column type
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            is_numeric = pd.api.types.is_numeric_dtype(series)

            # Try coercion for datetime if not detected
            if not is_numeric and not is_datetime:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    try:
                        series_converted = pd.to_datetime(series, errors="coerce")
                    except Exception:
                        series_converted = series
                    if series_converted.notna().sum() > 0 and series_converted.dtype != "object":
                        series = series_converted
                        is_datetime = True

            if not is_numeric and not is_datetime:
                raise LogLoadError(
                    f"Column '{column}' must be numeric or datetime for statistical deviation filtering."
                )

            def compute_bounds(sub_series: pd.Series) -> tuple[pd.Timestamp | float, pd.Timestamp | float]:
                """
                    Internal helper to compute deviation bounds based on series type.
                    Returns lower and upper bounds for comparison.
                    """
                mean_value = sub_series.mean()
                if is_datetime:
                    # Convert deviation_percent into equivalent timedelta based on total range
                    total_seconds = (sub_series.max() - sub_series.min()).total_seconds()
                    deviation_seconds = (deviation_percent / 100.0) * total_seconds
                    lower_bound = mean_value - pd.to_timedelta(deviation_seconds, unit="s")
                    upper_bound = mean_value + pd.to_timedelta(deviation_seconds, unit="s")
                else:
                    deviation_value = (deviation_percent / 100.0) * mean_value
                    lower_bound = mean_value - deviation_value
                    upper_bound = mean_value + deviation_value
                return lower_bound, upper_bound

            # Prepare mask of outliers
            if groupby_column and groupby_column in df.columns:
                masks = []
                for _, sub_df in df.groupby(groupby_column):
                    lower, upper = compute_bounds(sub_df[column])
                    mask_sub = (sub_df[column] < lower) | (sub_df[column] > upper)
                    masks.append(mask_sub)
                outside_mask = pd.concat(masks, axis=0).sort_index()
            else:
                lower, upper = compute_bounds(series)
                outside_mask = (series < lower) | (series > upper)

            # Apply chosen action
            if on_outside == "delete":
                df = df.loc[~outside_mask]
            elif on_outside == "nan":
                df.loc[outside_mask, column] = pd.NA
            elif on_outside == "default":
                fill_val = default_value if default_value is not None else series.mean()
                df.loc[outside_mask, column] = fill_val
            elif on_outside == "medium":
                df.loc[outside_mask, column] = series.median()
            elif on_outside == "max":
                df.loc[outside_mask, column] = series.max()
            elif on_outside == "min":
                df.loc[outside_mask, column] = series.min()
            elif on_outside == "show":
                if getattr(self, "debug", False):
                    print(f"[DEBUG] Outliers for '{column}' → {df.loc[outside_mask, column].tolist()}")

            # Update inplace if needed
            if inplace:
                self.df = df

            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] filter_by_statistical_deviation() failed: {e}")
            raise LogLoadError(f"Failed statistical deviation filtering for '{column}'.") from None

    def sort_by_columns(
            self,
            columns: Union[str, list[str], None] = None,
            ascending: Union[bool, list[bool]] = True,
            inplace: bool = False,
            na_position: Literal["first", "last"] = "last"
    ) -> pd.DataFrame:
        """
        Sort the DataFrame by one or multiple columns.

        Parameters
        ----------
        columns : str, list of str, or None, default=None
            Column name(s) to sort by. If None is provided,
            it defaults to ['case:concept:name', 'time:timestamp'].
        ascending : bool or list of bool, default=True
            Sort order (True = ascending). If list is provided, must match columns length.
        inplace : bool, default=False
            If False, returns a sorted copy of DataFrame without modifying self.df.
        na_position : {'first', 'last'}, default='last'
            Position of NaN values in sorted result.

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame by the specified or default column(s).

        Raises
        ------
        LogLoadError
            If the given column(s) do not exist or sorting fails.
        """

        try:
            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame found in instance.")

            df = self.df.copy()

            # 🧩 If no columns specified, apply default sorting logic
            if columns is None:
                default_cols = ["case:concept:name", "time:timestamp"]
                columns = [c for c in default_cols if c in df.columns]

                if not columns:
                    raise LogLoadError(
                        "No columns provided and default sort columns not found in DataFrame."
                    )

            # normalize input to list for uniform handling
            if isinstance(columns, str):
                columns = [columns]

            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                raise LogLoadError(f"Column(s) not found: {missing_cols}")

            # perform sorting
            sorted_df = df.sort_values(by=columns, ascending=ascending, na_position=na_position)

            if inplace:
                self.df = sorted_df
                return self.df
            else:
                return sorted_df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] sort_by_columns() failed: {e}")
            raise LogLoadError(f"Failed to sort DataFrame by columns '{columns}'.") from None

    def describe(self, summary_only: bool = False) -> dict:
        """
        Produce a comprehensive statistical and structural summary of the dataset.

        Parameters
        ----------
        summary_only : bool, default=False
            If True, returns only high-level sections:
            - summary
            - event_log_metrics
            - data_quality
            If False, includes all detailed column-level statistics.

        Returns
        -------
        dict
            A structured dictionary containing dataset summaries, column-level
            statistics (numeric, datetime, categorical), event log metrics, and
            data quality diagnostics.

        Raises
        ------
        LogLoadError
            If DataFrame is missing or no valid columns are found.
        LogFormatError
            If datetime or numeric parsing encounters inconsistent or invalid data.
        """
        try:
            import numpy as np
            import pandas as pd
            import warnings

            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame found.")

            df = self.df.copy()
            warnings.filterwarnings("ignore", category=UserWarning)

            # -------------------------------
            # Section 1: Summary
            # -------------------------------
            summary_section = {
                "description": "High-level dataset structure and metadata.",
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "column_types": {
                    "numeric": int(df.select_dtypes(include=["number"]).shape[1]),
                    "datetime": int(df.select_dtypes(include=["datetime"]).shape[1]),
                    "categorical": int(df.select_dtypes(include=["object", "category"]).shape[1]),
                    "boolean": int(df.select_dtypes(include=["bool"]).shape[1]),
                },
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
            }

            # If summary_only → return only summary + core metrics
            # (event_log_metrics + data_quality)
            # We compute all here so structure is identical
            # but return short version later.
            # ----------------------------------------------------

            # -------------------------------
            # Section 2: Numeric Columns
            # -------------------------------
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            numeric_section = {
                "description": "Statistical summary for numerical columns.",
                "columns": {}
            }

            for col in numeric_cols:
                col_series = df[col]
                desc = col_series.describe()

                q1 = col_series.quantile(0.25)
                q3 = col_series.quantile(0.75)
                iqr = float(q3 - q1)
                outlier_mask = (col_series < q1 - 1.5 * iqr) | (col_series > q3 + 1.5 * iqr)

                numeric_section["columns"][col] = {
                    "count": float(desc.get("count", 0)),
                    "mean": float(desc.get("mean", np.nan)),
                    "std": float(desc.get("std", np.nan)),
                    "min": float(desc.get("min", np.nan)),
                    "25%": float(desc.get("25%", np.nan)),
                    "50%": float(desc.get("50%", np.nan)),
                    "75%": float(desc.get("75%", np.nan)),
                    "max": float(desc.get("max", np.nan)),
                    "iqr": float(iqr),
                    "outliers_count": int(outlier_mask.sum()),
                    "n_missing": int(col_series.isna().sum()),
                    "missing_percent": float(col_series.isna().mean() * 100)
                }

            # -------------------------------
            # Section 3: Datetime Columns
            # -------------------------------
            datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
            datetime_section = {
                "description": "Temporal properties and distribution of datetime columns.",
                "columns": {}
            }

            for col in datetime_cols:
                s = df[col].dropna().sort_values()

                if len(s) == 0:
                    datetime_section["columns"][col] = {
                        "min": None,
                        "max": None,
                        "total_span_seconds": None,
                        "unique_timestamps": 0,
                        "min_delta_seconds": None,
                        "max_delta_seconds": None,
                        "mean_delta_seconds": None,
                        "n_missing": int(df[col].isna().sum()),
                        "missing_percent": float(df[col].isna().mean() * 100)
                    }
                    continue

                deltas = s.diff().dropna()
                total_span = (s.max() - s.min()).total_seconds()

                datetime_section["columns"][col] = {
                    "min": str(s.min()),
                    "max": str(s.max()),
                    "total_span_seconds": float(total_span),
                    "unique_timestamps": int(s.nunique()),
                    "min_delta_seconds": float(deltas.min().total_seconds()) if len(deltas) else None,
                    "max_delta_seconds": float(deltas.max().total_seconds()) if len(deltas) else None,
                    "mean_delta_seconds": float(deltas.mean().total_seconds()) if len(deltas) else None,
                    "n_missing": int(df[col].isna().sum()),
                    "missing_percent": float(df[col].isna().mean() * 100)
                }

            # -------------------------------
            # Section 4: Categorical Columns
            # -------------------------------
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            categorical_section = {
                "description": "Frequency, entropy, and top-value summary for categorical columns.",
                "columns": {}
            }

            for col in categorical_cols:
                s = df[col].dropna()
                value_counts = s.value_counts(normalize=False)
                total = len(df)

                if len(value_counts) > 0:
                    top_val = value_counts.index[0]
                    top_freq = int(value_counts.iloc[0])
                else:
                    top_val = None
                    top_freq = 0

                probs = value_counts / value_counts.sum() if len(value_counts) > 0 else []
                entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if len(probs) else 0.0

                top10 = [
                    {
                        "value": str(idx),
                        "count": int(cnt),
                        "percent": float((cnt / total) * 100)
                    }
                    for idx, cnt in value_counts.head(10).items()
                ]

                categorical_section["columns"][col] = {
                    "unique_count": int(s.nunique()),
                    "top_value": str(top_val) if top_val is not None else None,
                    "top_freq": top_freq,
                    "top_percent": float((top_freq / total) * 100),
                    "entropy": float(entropy),
                    "n_missing": int(df[col].isna().sum()),
                    "missing_percent": float(df[col].isna().mean() * 100),
                    "top_10_values": top10
                }

            # -------------------------------
            # Section 5: Event Log Metrics
            # -------------------------------
            # Only if standard xes naming exists
            case_col = "case:concept:name"
            time_col = "time:timestamp"
            act_col = "concept:name"
            res_col = "org:resource"

            event_section = {
                "description": "Event log behavioral and structural metrics.",
                "case_metrics": {},
                "throughput_time": {},
                "activity_metrics": {},
                "resource_metrics": {}
            }

            if case_col in df.columns:
                case_counts = df[case_col].value_counts()
                event_section["case_metrics"] = {
                    "n_cases": int(case_counts.shape[0]),
                    "mean_events_per_case": float(case_counts.mean()),
                    "median_events_per_case": float(case_counts.median()),
                    "min_events_per_case": int(case_counts.min()),
                    "max_events_per_case": int(case_counts.max())
                }

            if time_col in df.columns and case_col in df.columns:
                times = df[[case_col, time_col]].dropna().sort_values([case_col, time_col])

                durations = (
                    times.groupby(case_col)[time_col]
                    .agg(lambda x: (x.max() - x.min()).total_seconds())
                )

                if len(durations) > 0:
                    event_section["throughput_time"] = {
                        "mean_case_duration_seconds": float(durations.mean()),
                        "median_case_duration_seconds": float(durations.median()),
                        "min_case_duration_seconds": float(durations.min()),
                        "max_case_duration_seconds": float(durations.max())
                    }

            if act_col in df.columns:
                counts = df[act_col].value_counts()
                total = len(df)
                event_section["activity_metrics"] = {
                    "n_unique_activities": int(counts.shape[0]),
                    "top_activities": [
                        {
                            "activity": str(idx),
                            "count": int(cnt),
                            "percent": float((cnt / total) * 100)
                        }
                        for idx, cnt in counts.head(10).items()
                    ],
                    "rare_activities": [
                        {
                            "activity": str(idx),
                            "count": int(cnt),
                            "percent": float((cnt / total) * 100)
                        }
                        for idx, cnt in counts[counts == 1].items()
                    ]
                }

            if res_col in df.columns:
                counts = df[res_col].value_counts()
                total = len(df)
                event_section["resource_metrics"] = {
                    "n_unique_resources": int(counts.shape[0]),
                    "top_resources": [
                        {
                            "resource": str(idx),
                            "count": int(cnt),
                            "percent": float((cnt / total) * 100)
                        }
                        for idx, cnt in counts.head(10).items()
                    ]
                }

            # -------------------------------
            # Section 6: Data Quality
            # -------------------------------
            data_quality_section = {
                "description": "Data integrity diagnostics including missingness, duplicates, skewness, and anomalies.",
                "missing_values": {},
                "duplicate_rows": {},
                "constant_columns": [],
                "high_cardinality_columns": [],
                "skewed_columns": {}
            }

            # Missing values
            missing_total = int(df.isna().sum().sum())
            missing_percent = float((df.isna().sum().sum() / (df.size)) * 100)

            data_quality_section["missing_values"] = {
                "total_missing_cells": missing_total,
                "missing_percent": missing_percent,
                "columns_with_missing": {
                    col: int(df[col].isna().sum())
                    for col in df.columns
                    if df[col].isna().sum() > 0
                }
            }

            # Duplicates
            dup_count = int(df.duplicated().sum())
            data_quality_section["duplicate_rows"] = {
                "duplicate_count": dup_count,
                "duplicate_percent": float((dup_count / len(df)) * 100)
            }

            # Constant columns
            data_quality_section["constant_columns"] = [
                col for col in df.columns if df[col].nunique() <= 1
            ]

            # High cardinality
            data_quality_section["high_cardinality_columns"] = [
                col for col in df.columns if df[col].nunique() > 0.8 * len(df)
            ]

            # Skewness for numeric columns
            for col in numeric_cols:
                try:
                    data_quality_section["skewed_columns"][col] = {
                        "skew_value": float(df[col].skew())
                    }
                except Exception:
                    pass

            # -------------------------------
            # Build final result
            # -------------------------------
            if summary_only:
                return {
                    "summary": summary_section,
                    "event_log_metrics": event_section,
                    "data_quality": data_quality_section
                }

            return {
                "summary": summary_section,
                "numeric_columns": numeric_section,
                "datetime_columns": datetime_section,
                "categorical_columns": categorical_section,
                "event_log_metrics": event_section,
                "data_quality": data_quality_section
            }

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] describe() failed: {e}")
            raise LogLoadError("Failed to generate description summary.") from None

    def merge_columns(
            self,
            columns: list[str],
            new_column: str,
            delimiter: str = "_",
            drop_original: bool = False,
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Merge multiple columns into a single new column using a chosen delimiter.

        Parameters
        ----------
        columns : list of str
            Names of the columns to merge.
        new_column : str
            Name of the resulting merged column.
        delimiter : str, default="_"
            Character or string inserted between concatenated values.
        drop_original : bool, default=False
            If True, removes the original columns after merging.
            Defaults to False (safe mode, keeps originals).
        inplace : bool, default=False
            If True, modifies self.df directly and returns it.
            Otherwise, returns a copy of the DataFrame with changes.

        Returns
        -------
        pd.DataFrame
            DataFrame with the new merged column added.

        Raises
        ------
        LogLoadError
            If any of the specified columns do not exist.
        LogFormatError
            If merging fails due to inconsistent or invalid data types.
        """
        try:
            import pandas as pd

            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame available in instance.")

            df = self.df.copy()

            # Validate column existence
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise LogLoadError(f"Column(s) not found: {missing}")

            if len(columns) < 2:
                raise LogLoadError("At least two columns must be provided for merging.")

            # Convert all selected columns to string and concatenate
            try:
                df[new_column] = df[columns].astype(str).agg(delimiter.join, axis=1)
            except Exception as e:
                raise LogFormatError(f"Failed to merge columns: {e}")

            # Drop originals if requested
            if drop_original:
                df.drop(columns=columns, inplace=True, errors="ignore")

            # Apply inplace logic
            if inplace:
                self.df = df
                return self.df
            return df

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] merge_columns() failed: {e}")
            raise LogLoadError(f"Failed to merge columns: {e}") from None

    def suggest_corrections(
            self,
            column: str,
            valid_values: list[str],
            threshold: int = 85
    ) -> pd.DataFrame:
        """
        Suggest fuzzy corrections for textual values in a column
        based on a provided list of valid terms.

        Parameters
        ----------
        column : str
            Name of the column to analyze for textual mismatches.
        valid_values : list of str
            The list of valid, accepted reference values.
        threshold : int, default=85
            Minimum similarity ratio for a match to be considered a suggestion (0–100).

        Returns
        -------
        pd.DataFrame
            A DataFrame showing invalid values and their suggested corrections,
            including similarity scores.

        Raises
        ------
        LogLoadError
            If DataFrame is missing or column not found.
        LogFormatError
            If fuzzy matching fails due to invalid data types.
        """
        try:

            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame available in instance.")

            if column not in self.df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            s = self.df[column].dropna().astype(str)
            unique_vals = s.unique().tolist()
            suggestions = []

            for val in unique_vals:
                match = process.extractOne(
                    val,
                    valid_values,
                    scorer=fuzz.WRatio
                )
                if match and match[1] >= threshold:
                    # Only include if the suggested value isn't exactly the same
                    if match[0] != val:
                        suggestions.append({
                            "original_value": val,
                            "suggested_value": match[0],
                            "similarity_score": float(match[1])
                        })

            return pd.DataFrame(suggestions)

        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[ERROR] suggest_corrections() failed: {e}")
            raise LogFormatError(f"Failed to compute fuzzy suggestions: {e}") from None

    def apply_corrections(
            self,
            column: str,
            corrections_df,
            inplace: bool = False
    ) -> pd.DataFrame:
        """
        Apply approved textual corrections to a specified column of the DataFrame.

        This method accepts either a DataFrame (the output of `suggest_corrections()`)
        or a dictionary mapping of incorrect → corrected values, applies normalization
        (strip + uppercase) to both sides to ensure exact matches, and returns a corrected
        DataFrame copy unless `inplace=True` is specified.

        Parameters
        ----------
        column : str
            Name of the target column in which corrections will be applied.
        corrections_df : pd.DataFrame or dict
            Correction source. If DataFrame, must include columns
            ['original_value', 'suggested_value'] from `suggest_corrections()`.
            If dict, keys represent wrong values and values the corrected ones.
        inplace : bool, default=False
            Whether to modify `self.df` directly. When False, works on a copy.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with normalized and corrected text values.

        Raises
        ------
        LogLoadError
            If DataFrame or target column is missing.
        LogFormatError
            If the correction format or data normalization fails.
        """
        try:

            if getattr(self, "df", None) is None:
                raise LogLoadError("No active DataFrame available in instance.")

            if column not in self.df.columns:
                raise LogLoadError(f"Column '{column}' not found in DataFrame.")

            # work on safe copy to preserve immutability
            df = self.df.copy()

            # normalize column
            df[column] = df[column].astype(str).str.strip().str.upper()

            # determine correction mapping
            try:
                if isinstance(corrections_df, pd.DataFrame):
                    print("in if")
                    required_cols = {"original_value", "suggested_value"}
                    if not required_cols.issubset(corrections_df.columns):
                        raise LogFormatError(
                            "Input DataFrame must contain 'original_value' and 'suggested_value' columns."
                        )

                    corrections_map = dict(
                        zip(
                            corrections_df["original_value"].astype(str).str.strip().str.upper(),
                            corrections_df["suggested_value"].astype(str).str.strip().str.upper(),
                        )
                    )
                elif isinstance(corrections_df, dict):
                    print("in elif")
                    # normalize dictionary keys and values
                    corrections_map = {
                        str(k).strip().upper(): str(v).strip().upper()
                        for k, v in corrections_df.items()
                    }
                else:
                    raise LogFormatError(
                        "Corrections must be provided either as a dict or as a DataFrame from suggest_corrections()."
                    )
            except Exception as e:
                raise LogFormatError(f"Failed to interpret corrections mapping: {e}")

            # apply corrections
            try:
                df[column] = df[column].replace(corrections_map)
            except Exception as e:
                raise LogFormatError(f"Failed during DataFrame replacement: {e}")

            # respect `inplace` behavior
            if inplace:
                self.df = df
                return self.df
            return df

        except (LogLoadError, LogFormatError):
            raise
        except Exception as e:
            raise LogLoadError(f"Unexpected error in apply_corrections(): {e}") from None

    def smart_clean(
            self,
            aggressive: bool = False,
            infer_types: bool = True,
            normalize_names: bool = True,
            scope: Literal["selected", "all", "xes"] = "selected",
            inplace: bool = True
    ) -> pd.DataFrame:
        """
        Simplified Smart Clean:
        - Performs full cleaning steps ONLY when inplace=True.
        - When inplace=False: does nothing and simply returns a copy.

        All other methods in the class remain untouched.
        """

        # -------------------------------------------------
        # 0) Validation
        # -------------------------------------------------
        if getattr(self, "df", None) is None:
            raise LogLoadError("No DataFrame available in this LogPreprocessor instance.")

        # If inplace=False → return a safe copy without doing anything
        if not inplace:
            self._log_info("smart_clean called with inplace=False → No cleaning performed.")
            return self.df.copy()

        #try:
        df = self.df.copy()

        # -------------------------------------------------
        # 1) Determine active columns
        # -------------------------------------------------
        xes_standard = [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "org:resource",
            "lifecycle:transition"
        ]

        if scope == "selected":
            active_cols = [c for c in getattr(self, "selected_columns", xes_standard) if c in df.columns]
        elif scope == "xes":
            active_cols = [c for c in xes_standard if c in df.columns]
        else:
            active_cols = df.columns.tolist()

        self._log_info(f"[Scope={scope}] Active columns for cleaning → {active_cols}")

        # -------------------------------------------------
        # 2) Normalize column names
        # -------------------------------------------------
        if normalize_names:
            normalized = [c.strip().lower().replace(" ", "_") for c in df.columns]
            df.columns = normalized
            self._log_info("Normalized column names to lowercase_underscore style.")

        # -------------------------------------------------
        # 3) Drop constant / high-null columns
        # (Your upgraded version already supports null_threshold)
        # -------------------------------------------------
        df = self.drop_constant_columns(null_threshold=0.9, inplace=True)

        # Remove fully empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = self.drop_columns(columns=empty_cols, inplace=True)
            self._log_info(f"Dropped fully empty columns: {empty_cols}")

        # -------------------------------------------------
        # 4) Type inference
        # -------------------------------------------------
        if infer_types:
            for col in active_cols:
                if col not in df.columns:
                    continue

                if "time" in col and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df = self.convert_column_type(col, "datetime", inplace=True)
                    self._log_info(f"Converted {col} → datetime")

                elif pd.api.types.is_object_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col])
                        self._log_info(f"Auto‑typed {col} → numeric")
                    except Exception:
                        if df[col].nunique() < 50:
                            df[col] = df[col].astype("category")
                            self._log_info(f"Auto‑typed {col} → category")

        # -------------------------------------------------
        # 5) Handle nulls in critical XES columns
        # -------------------------------------------------
        criticals = [c for c in ["case:concept:name", "concept:name", "time:timestamp"] if c in df.columns]
        for c in criticals:
            df = self.handle_null_values(column=c, on_null="delete", inplace=True)
            self._log_info(f"Deleted NaN values in critical column: {c}")

        # -------------------------------------------------
        # 6) Remove duplicates (global)
        # -------------------------------------------------
        before = len(df)
        df = self.remove_duplicates(inplace=True)
        removed = before - len(df)
        if removed > 0:
            self._log_info(f"Removed {removed} duplicate rows.")

        # -------------------------------------------------
        # 7) Sort by case + time (IEEE semantics)
        # -------------------------------------------------
        df = self.sort_by_columns(columns=["case:concept:name", "time:timestamp"], inplace=True)
        self._log_info("Sorted by [case:concept:name, time:timestamp].")

        # -------------------------------------------------
        # 8) Outlier removal (optional)
        # -------------------------------------------------
        if aggressive and "time:timestamp" in df.columns:
            try:
                df = self.handle_outliers(column="time:timestamp", inplace=True)
                self._log_info("Applied aggressive outlier filtering (IQR).")
            except Exception as ex:
                self._log_info(f"Outlier filtering skipped: {ex}")


        df = self.anonymize_column("case:concept:name", mode="numeric", inplace=True)
        df = self.drop_columns(["concept:name", "time:timestamp", "case:concept:name"], keep_mode=True)

        # -------------------------------------------------
        # Finalize: always inplace=True
        # -------------------------------------------------
        self.df = df
        self._log_info("✅ smart_clean completed successfully (inplace updated).")
        self.df.to_csv("dataframe_cleaned.csv")
        return self.df

        # except Exception as e:
        #     if getattr(self, "debug", False):
        #         print(f"[ERROR] smart_clean() failed: {e}")
        #     raise LogLoadError(f"smart_clean() failed: {e}") from None


class LogFileNotFoundError(IOError):
    """Raised when the target log file does not exist."""
    pass


class LogFormatError(ValueError):
    """Raised when file format is unsupported or corrupted."""
    pass


class LogLoadError(RuntimeError):
    """Raised when loading process fails for any internal reason."""
    pass
