# Pandas FutureWarnings Fix Summary

This document summarizes the fixes applied to resolve pandas FutureWarnings in the cost-model codebase.

## Overview

The fixes address two main categories of pandas FutureWarnings:

1. **Downcasting warnings with fillna operations**
2. **DataFrame concatenation with empty DataFrames**

## 1. Downcasting Warnings Fix

### Problem
Pandas was issuing FutureWarnings when using patterns like:
```python
series.fillna(value).astype(dtype)
```

The warning message was:
```
FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead.
```

### Solution
Use `infer_objects(copy=False)` **before** calling `fillna()`:

**Before (causing warnings):**
```python
numeric_series = pd.to_numeric(str_series, errors='coerce').fillna(0)
result = numeric_series.astype('int64')
```

**After (fixed):**
```python
numeric_series = pd.to_numeric(str_series, errors='coerce')
inferred_series = numeric_series.infer_objects(copy=False)
result = inferred_series.fillna(0).astype('int64')
```

### Files Modified

#### `cost_model/engines/run_one_year/finalize.py`
- **Line 416-419**: Fixed integer conversion pattern in `finalize_snapshot()`
- **Change**: Added `infer_objects(copy=False)` before `fillna(0)` for int64 dtype conversion

#### `cost_model/projections/hazard.py`
- **Line 227-230**: Fixed numeric conversion for rate columns
- **Change**: Added `infer_objects(copy=False)` before `fillna(0.0)` for float conversion

## 2. DataFrame Concatenation Warnings Fix

### Problem
Pandas was issuing FutureWarnings when concatenating DataFrames that might include empty ones:
```
FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
```

### Solution
Filter out empty DataFrames before concatenation:

**Before (causing warnings):**
```python
result = pd.concat([df1, df2, empty_df], ignore_index=True)
```

**After (fixed):**
```python
dfs_to_concat = [df for df in [df1, df2, empty_df] if not df.empty]
if dfs_to_concat:
    result = pd.concat(dfs_to_concat, ignore_index=True)
else:
    result = pd.DataFrame(columns=expected_columns)
```

### Files Modified

#### `cost_model/engines/run_one_year/orchestrator/__init__.py`
- **Line 408-413**: Added filtering for event consolidation
- **Line 479-485**: Added filtering for event log concatenation
- **Change**: Filter empty DataFrames before `pd.concat()` operations

#### `cost_model/state/snapshot_update.py`
- **Line 262-269**: Added filtering for snapshot updates
- **Change**: Filter empty DataFrames before concatenation with fallback handling

## 3. Testing

Created comprehensive test suite in `test_pandas_warnings_fix.py` that verifies:

1. ✅ `fillna().astype()` pattern with `infer_objects()`
2. ✅ `pd.concat()` with empty DataFrame filtering
3. ✅ Extension array dtype conversions
4. ✅ Numeric conversion patterns

All tests pass with no FutureWarnings detected.

## 4. Benefits

- **Future Compatibility**: Code will continue to work with future pandas versions
- **Clean Warnings**: No more FutureWarnings cluttering logs
- **Best Practices**: Follows pandas recommended patterns
- **Performance**: `copy=False` parameter optimizes memory usage

## 5. Key Principles Applied

1. **Use `infer_objects(copy=False)` before `fillna()`** for dtype operations
2. **Filter empty DataFrames before `pd.concat()`** operations
3. **Provide fallback handling** when all DataFrames are empty
4. **Test thoroughly** to ensure no regressions

## 6. Verification

Run the test suite to verify fixes:
```bash
python test_pandas_warnings_fix.py
```

Expected output:
```
🎉 All pandas FutureWarning fixes are working correctly!
```

## 7. Future Maintenance

When adding new pandas operations:
- Use `infer_objects(copy=False)` before `fillna()` for dtype conversions
- Filter empty DataFrames before concatenation
- Test for FutureWarnings in development

This ensures the codebase remains compatible with future pandas versions.
