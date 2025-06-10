# cost_model/projections/snapshot/migration_safe.py
"""
Safe column migration that prevents duplicates.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SafeColumnMigrator:
    """Safely migrate columns without creating duplicates."""

    @staticmethod
    def migrate_columns(
        df: pd.DataFrame,
        mapping: Dict[str, str],
        remove_old: bool = True,
        on_conflict: str = "skip",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Safely migrate columns with duplicate prevention.

        Args:
            df: DataFrame to migrate
            mapping: Old to new column name mapping
            remove_old: Whether to remove old columns after migration
            on_conflict: What to do if new column already exists
                - "skip": Skip the migration for that column
                - "replace": Replace existing column
                - "merge": Merge with existing column (first non-null)

        Returns:
            Tuple of (migrated_df, list_of_warnings)
        """
        df = df.copy()
        warnings = []

        for old_col, new_col in mapping.items():
            if old_col not in df.columns:
                continue

            if new_col in df.columns and old_col != new_col:
                # Conflict detected
                if on_conflict == "skip":
                    warnings.append(f"Skipped migration {old_col}->{new_col}: target exists")
                    continue

                elif on_conflict == "replace":
                    warnings.append(f"Replacing existing column {new_col} with {old_col}")
                    df[new_col] = df[old_col]

                elif on_conflict == "merge":
                    # Merge: take first non-null value
                    warnings.append(f"Merging {old_col} into existing {new_col}")
                    df[new_col] = df[new_col].fillna(df[old_col])

            else:
                # No conflict, simple rename
                if old_col != new_col:
                    df[new_col] = df[old_col]

            # Remove old column if requested and it's different from new
            if remove_old and old_col != new_col and old_col in df.columns:
                df = df.drop(columns=[old_col])

        return df, warnings
