# Handling Duplicate Column Labels in Snapshot Creation

## 1. Problem Statement

The snapshot creation process has been intermittently failing due to the presence of duplicate column labels within pandas DataFrames at various stages of processing. These duplicates prevent certain pandas operations from executing as expected, leading to errors that halt the projection.

## 2. Root Cause Analysis

Duplicate column labels can originate from several sources:

*   **Initial Census Data**: The raw census input file might contain columns with identical names.
*   **Legacy Column Migration**: The `migrate_legacy_columns` utility, while designed to map old column names to new ones, might inadvertently result in both old and new names co-existing if not handled perfectly, or if the underlying data already had subtle duplications.
*   **DataFrame Concatenation/Merging**: Operations that combine DataFrames can introduce duplicate columns if not managed carefully.

## 3. Symptoms and Errors Encountered

The presence of duplicate columns has manifested in at least two distinct errors:

### 3.1. `TypeError: cannot convert the series to <class 'int'>`

*   **Context**: This error occurred during compensation normalization, specifically when trying to count missing compensation values: `missing_count = int(missing_comp.sum())`.
*   **Reason**: If a column name (e.g., `employee_gross_comp`) is duplicated, `df[comp_col]` returns a DataFrame instead of a Series. Consequently, `missing_comp.sum()` (where `missing_comp` is `df[comp_col].isna()`) also returns a Series (one sum per duplicate column), which cannot be directly cast to an `int`.

### 3.2. `ValueError: cannot assemble with duplicate keys`

*   **Context**: This error occurred during tenure calculation when converting the hire date column to datetime objects: `hire_dates = pd.to_datetime(df[hire_date_col])`.
*   **Reason**: Similar to the above, if `hire_date_col` (e.g., `EMP_HIRE_DATE`) is a duplicated label, `df[hire_date_col]` returns a DataFrame. `pd.to_datetime()` expects a Series or a scalar for this type of conversion and fails when passed a DataFrame with multiple columns of the same name.

## 4. Impact

These errors are critical as they prevent the successful creation of the initial employee snapshot, which is a foundational step for running any workforce projections.

## 5. Short-Term Solutions Implemented

To address these issues and allow projections to proceed, the following tactical fixes have been implemented:

1.  **Initial Deduplication in `SnapshotBuilder`**:
    *   The line `census_df = census_df.loc[:, ~census_df.columns.duplicated()]` was added to `SnapshotBuilder.create_initial_snapshot` immediately after the census data is loaded and processed by `census_processor.process_census_data`. This aims to remove any duplicates present in the initial dataset.

2.  **`_dedup_columns` Utility Function**:
    *   A utility function `_dedup_columns(df: pd.DataFrame) -> pd.DataFrame` was created in `cost_model/utils/frame_tools.py`. This function checks for duplicated columns and, if found, logs a warning and removes them, keeping the first occurrence.

3.  **Deduplication After Legacy Column Migration in `SnapshotTransformer`**:
    *   The `_dedup_columns` utility is now called immediately after `migrate_legacy_columns` in the following methods of `SnapshotTransformer` (`cost_model/projections/snapshot/transformers.py`):
        *   `apply_tenure_calculations`
        *   `apply_age_calculations`
    *   This ensures that any duplicates introduced or left unresolved by the migration process are cleaned up before subsequent operations that are sensitive to unique column names (like date conversions or aggregations).

## 6. Request for Long-Term Architectural Review

The implemented solutions are primarily reactive fixes to ensure the immediate functionality of the projection pipeline. A more robust, long-term architectural solution for column name management, schema enforcement, and migration is recommended. This could involve:

*   Stricter schema validation at data ingestion points.
*   A more sophisticated column migration strategy that guarantees uniqueness or explicitly handles intended duplications (if any).
*   Centralized DataFrame validation utilities that check for and resolve duplicate columns at critical checkpoints.

This would help prevent similar issues from arising in the future and improve the overall robustness and maintainability of the cost model.
