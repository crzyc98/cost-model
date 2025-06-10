import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated column labels while keeping the first occurrence."""
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        unique_dups = list(set(dups))
        logger.warning("Dropping duplicated columns: %s", unique_dups)
        
        # Log details for debugging
        for dup_col in unique_dups:
            count = list(df.columns).count(dup_col)
            logger.debug(f"Column '{dup_col}' appears {count} times")
        
        # Remove duplicates by keeping only the first occurrence
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        logger.debug(f"After deduplication: {len(df.columns)} columns remaining")
    return df
