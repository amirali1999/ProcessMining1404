from typing import List, Dict, Any, Optional
import pandas as pd
from prefixspan import PrefixSpan


class PatternMiningError(RuntimeError):
    """Raised when frequent pattern mining fails."""
    pass


class PatternMiner:
    """
    Mine frequent sequential patterns from process traces.

    This class operates on TRACE‑LEVEL data and is typically used
    after TraceAnalyzer.build_traces().
    """

    def __init__(
        self,
        traces: pd.Series,
        min_support: int = 2
    ) -> None:
        """
        Parameters
        ----------
        traces : pd.Series
            Index = case_id, value = list of activity names.
        min_support : int, default=2
            Minimum number of traces a pattern must appear in.
        """
        if not isinstance(traces, pd.Series):
            raise PatternMiningError("traces must be a pandas Series (case_id → list of activities).")

        self.traces = traces.tolist()
        self.min_support = min_support
        self.patterns = None

    # ------------------------------------------------------------------

    def mine_patterns(
        self,
        max_len: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run PrefixSpan algorithm to mine frequent sequential patterns.

        Parameters
        ----------
        max_len : int, optional
            Maximum length of patterns to mine.

        Returns
        -------
        pd.DataFrame
            Columns: ['pattern', 'support']
        """
        try:
            ps = PrefixSpan(self.traces)
            ps.minlen = 1
            if max_len:
                ps.maxlen = max_len

            raw_patterns = ps.frequent(self.min_support)

            df = pd.DataFrame(
                [
                    {
                        "pattern": tuple(patt),
                        "support": int(sup)
                    }
                    for sup, patt in raw_patterns
                ]
            ).sort_values(
                by=["support", "pattern"],
                ascending=[False, True]
            ).reset_index(drop=True)

            self.patterns = df
            return df

        except Exception as e:
            raise PatternMiningError(f"Pattern mining failed: {e}") from None

    # ------------------------------------------------------------------

    def top_patterns(
        self,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top‑N most frequent patterns.

        Parameters
        ----------
        n : int, default=10

        Returns
        -------
        pd.DataFrame
        """
        if self.patterns is None:
            raise PatternMiningError("Patterns not mined yet. Call mine_patterns() first.")

        return self.patterns.head(n)
