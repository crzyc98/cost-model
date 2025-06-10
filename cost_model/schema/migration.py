"""
Migration utilities for legacy column names and schema compatibility.

This module provides utilities to migrate from legacy column naming conventions
to the new unified schema system while maintaining backward compatibility.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .columns import EventColumns, SnapshotColumns

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a schema migration operation."""

    success: bool
    migrated_columns: Dict[str, str]  # old_name -> new_name
    unmapped_columns: List[str]
    warnings: List[str]
    errors: List[str]


class LegacyColumnMapper:
    """Handles mapping between legacy and new column names."""

    # Legacy to new column mappings for snapshots
    # Organized by priority: preferred mappings first, alternatives second
    LEGACY_SNAPSHOT_MAPPING: Dict[str, str] = {
        # Primary EMP_ prefix mappings (legacy schema)
        "EMP_ID": SnapshotColumns.EMP_ID,
        "EMP_HIRE_DATE": SnapshotColumns.EMP_HIRE_DATE,
        "EMP_BIRTH_DATE": SnapshotColumns.EMP_BIRTH_DATE,
        "EMP_GROSS_COMP": SnapshotColumns.EMP_GROSS_COMP,
        "EMP_TERM_DATE": SnapshotColumns.EMP_TERM_DATE,
        "EMP_ACTIVE": SnapshotColumns.EMP_ACTIVE,
        "EMP_DEFERRAL_RATE": SnapshotColumns.EMP_DEFERRAL_RATE,
        "EMP_TENURE": SnapshotColumns.EMP_TENURE,
        "EMP_TENURE_BAND": SnapshotColumns.EMP_TENURE_BAND,
        "EMP_AGE": SnapshotColumns.EMP_AGE,
        "EMP_AGE_BAND": SnapshotColumns.EMP_AGE_BAND,
        "EMP_LEVEL": SnapshotColumns.EMP_LEVEL,
        "EMP_LEVEL_SOURCE": SnapshotColumns.EMP_LEVEL_SOURCE,
        "EMP_EXITED": SnapshotColumns.EMP_EXITED,
        "EMP_STATUS_EOY": SnapshotColumns.EMP_STATUS_EOY,
        "EMP_CONTRIBUTION": SnapshotColumns.EMP_CONTRIBUTION,
        "EMPLOYER_CORE": SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION,
        "EMPLOYER_MATCH": SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION,
        "IS_ELIGIBLE": SnapshotColumns.IS_ELIGIBLE,
        "SIMULATION_YEAR": SnapshotColumns.SIMULATION_YEAR,
        "TERM_RATE": SnapshotColumns.TERM_RATE,
        "PROMOTION_RATE": SnapshotColumns.PROMOTION_RATE,

        # Alternative EMP_ prefix mappings (avoid duplicates with primary)
        "EMP_CONTR": SnapshotColumns.EMP_CONTRIBUTION,  # Alternative for EMP_CONTRIBUTION
        "EMP_EMPLOYEE_CONTRIB": SnapshotColumns.EMP_CONTRIBUTION,  # Legacy alternative
        "EMP_EMPLOYER_MATCH": SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION,  # Alternative
        "ACTIVE_STATUS": SnapshotColumns.EMP_ACTIVE,  # Alternative for EMP_ACTIVE

        # Descriptive name mappings (only if not conflicting with EMP_ prefix)
        "employee_id": SnapshotColumns.EMP_ID,
        "employee_hire_date": SnapshotColumns.EMP_HIRE_DATE,
        "employee_birth_date": SnapshotColumns.EMP_BIRTH_DATE,
        "employee_gross_compensation": SnapshotColumns.EMP_GROSS_COMP,
        "employee_termination_date": SnapshotColumns.EMP_TERM_DATE,
        "active": SnapshotColumns.EMP_ACTIVE,
        "employee_deferral_rate": SnapshotColumns.EMP_DEFERRAL_RATE,
        "employee_tenure": SnapshotColumns.EMP_TENURE,
        "employee_tenure_band": SnapshotColumns.EMP_TENURE_BAND,
        "employee_age": SnapshotColumns.EMP_AGE,
        "employee_age_band": SnapshotColumns.EMP_AGE_BAND,
        "employee_level": SnapshotColumns.EMP_LEVEL,
        "job_level_source": SnapshotColumns.EMP_LEVEL_SOURCE,
        "exited": SnapshotColumns.EMP_EXITED,
        "employee_status_eoy": SnapshotColumns.EMP_STATUS_EOY,
        "employee_contribution": SnapshotColumns.EMP_CONTRIBUTION,
        "employer_core_contribution": SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION,
        "employer_match_contribution": SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION,
        "is_eligible": SnapshotColumns.IS_ELIGIBLE,
        "is_enrolled": SnapshotColumns.IS_ENROLLED,
        "simulation_year": SnapshotColumns.SIMULATION_YEAR,
        "term_rate": SnapshotColumns.TERM_RATE,
        "promotion_rate": SnapshotColumns.PROMOTION_RATE,
    }

    # Legacy to new column mappings for events
    LEGACY_EVENT_MAPPING: Dict[str, str] = {
        # Event schema mappings
        "event_id": EventColumns.EVENT_ID,
        "event_type": EventColumns.EVENT_TYPE,
        "event_date": EventColumns.EVENT_DATE,
        "employee_id": EventColumns.EMP_ID,
        "EMP_ID": EventColumns.EMP_ID,
        "gross_compensation": EventColumns.GROSS_COMPENSATION,
        "deferral_rate": EventColumns.DEFERRAL_RATE,
        "job_level": EventColumns.JOB_LEVEL,
        "termination_reason": EventColumns.TERMINATION_REASON,
        "simulation_year": EventColumns.SIMULATION_YEAR,
        "SIMULATION_YEAR": EventColumns.SIMULATION_YEAR,
        "event_payload": EventColumns.EVENT_PAYLOAD,
        "created_by": EventColumns.CREATED_BY,
        "created_at": EventColumns.CREATED_AT,
        "updated_at": EventColumns.UPDATED_AT,
    }

    # Common pattern mappings for fuzzy matching
    PATTERN_MAPPINGS: Dict[str, str] = {
        # Patterns that can be matched with regex or fuzzy logic
        r".*emp.*id.*": SnapshotColumns.EMP_ID,
        r".*hire.*date.*": SnapshotColumns.EMP_HIRE_DATE,
        r".*birth.*date.*": SnapshotColumns.EMP_BIRTH_DATE,
        r".*compensation.*": SnapshotColumns.EMP_GROSS_COMP,
        r".*term.*date.*": SnapshotColumns.EMP_TERM_DATE,
        r".*active.*": SnapshotColumns.EMP_ACTIVE,
        r".*deferral.*rate.*": SnapshotColumns.EMP_DEFERRAL_RATE,
        r".*tenure.*": SnapshotColumns.EMP_TENURE,
        r".*age.*": SnapshotColumns.EMP_AGE,
        r".*level.*": SnapshotColumns.EMP_LEVEL,
    }

    @classmethod
    def get_snapshot_mapping(cls, columns: List[str]) -> Dict[str, str]:
        """Get column mapping for snapshot data.

        Args:
            columns: List of column names to map

        Returns:
            Dictionary mapping old column names to new names
        """
        mapping = {}
        for col in columns:
            if col in cls.LEGACY_SNAPSHOT_MAPPING:
                mapping[col] = cls.LEGACY_SNAPSHOT_MAPPING[col]
        return mapping

    @classmethod
    def get_event_mapping(cls, columns: List[str]) -> Dict[str, str]:
        """Get column mapping for event data.

        Args:
            columns: List of column names to map

        Returns:
            Dictionary mapping old column names to new names
        """
        mapping = {}
        for col in columns:
            if col in cls.LEGACY_EVENT_MAPPING:
                mapping[col] = cls.LEGACY_EVENT_MAPPING[col]
        return mapping

    @classmethod
    def fuzzy_match_columns(cls, columns: List[str]) -> Dict[str, str]:
        """Attempt fuzzy matching for unmapped columns.

        Args:
            columns: List of column names to fuzzy match

        Returns:
            Dictionary mapping columns to best guesses
        """
        import re

        fuzzy_mapping = {}

        for col in columns:
            col_lower = col.lower()
            for pattern, target_col in cls.PATTERN_MAPPINGS.items():
                if re.match(pattern, col_lower):
                    fuzzy_mapping[col] = target_col
                    break

        return fuzzy_mapping

    @classmethod
    def get_reverse_mapping(cls, schema_type: str = "snapshot") -> Dict[str, str]:
        """Get reverse mapping (new -> old) for backward compatibility.

        Args:
            schema_type: Either "snapshot" or "event"

        Returns:
            Dictionary mapping new column names to legacy names
        """
        if schema_type == "snapshot":
            source_mapping = cls.LEGACY_SNAPSHOT_MAPPING
        elif schema_type == "event":
            source_mapping = cls.LEGACY_EVENT_MAPPING
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        # Create reverse mapping, preferring the first legacy name for each new name
        reverse_mapping = {}
        for old_name, new_name in source_mapping.items():
            if new_name not in reverse_mapping:
                reverse_mapping[new_name] = old_name

        return reverse_mapping


def migrate_legacy_columns(
    df: pd.DataFrame,
    schema_type: str = "snapshot",
    strict_mode: bool = False,
    attempt_fuzzy_match: bool = True,
) -> Tuple[pd.DataFrame, MigrationResult]:
    """Migrate a DataFrame from legacy column names to new schema.

    Args:
        df: DataFrame to migrate
        schema_type: Type of schema ("snapshot" or "event")
        strict_mode: If True, fail on any unmapped columns
        attempt_fuzzy_match: If True, attempt fuzzy matching for unmapped columns

    Returns:
        Tuple of (migrated_dataframe, migration_result)
    """
    result = MigrationResult(
        success=True, migrated_columns={}, unmapped_columns=[], warnings=[], errors=[]
    )

    # Get appropriate mapping
    if schema_type == "snapshot":
        column_mapping = LegacyColumnMapper.get_snapshot_mapping(df.columns.tolist())
    elif schema_type == "event":
        column_mapping = LegacyColumnMapper.get_event_mapping(df.columns.tolist())
    else:
        result.errors.append(f"Unknown schema type: {schema_type}")
        result.success = False
        return df, result

    # Identify unmapped columns
    mapped_columns = set(column_mapping.keys())
    all_columns = set(df.columns)
    unmapped_columns = all_columns - mapped_columns

    # Attempt fuzzy matching if enabled
    fuzzy_mapping = {}
    if attempt_fuzzy_match and unmapped_columns:
        fuzzy_mapping = LegacyColumnMapper.fuzzy_match_columns(list(unmapped_columns))
        for old_col, new_col in fuzzy_mapping.items():
            result.warnings.append(f"Fuzzy matched column '{old_col}' -> '{new_col}'")
        column_mapping.update(fuzzy_mapping)
        unmapped_columns = unmapped_columns - set(fuzzy_mapping.keys())

    # Handle unmapped columns
    if unmapped_columns:
        result.unmapped_columns = list(unmapped_columns)
        if strict_mode:
            result.errors.append(f"Unmapped columns in strict mode: {unmapped_columns}")
            result.success = False
            return df, result
        else:
            result.warnings.append(f"Unmapped columns (will be preserved): {unmapped_columns}")

    # Perform the migration
    try:
        migrated_df = df.copy()

        # Rename columns
        if column_mapping:
            # Check for potential duplicate target columns before renaming
            target_columns = list(column_mapping.values())
            target_counts = {}
            for target in target_columns:
                target_counts[target] = target_counts.get(target, 0) + 1

            duplicated_targets = {k: v for k, v in target_counts.items() if v > 1}
            if duplicated_targets:
                logger.debug(
                    f"Multiple source columns map to the same target: {duplicated_targets}"
                )
                logger.debug("Resolving duplicates by keeping first mapping for each target")

                # Remove columns that would create duplicates - keep the first mapping
                filtered_mapping = {}
                seen_targets = set()
                for old_col, new_col in column_mapping.items():
                    if new_col not in seen_targets:
                        filtered_mapping[old_col] = new_col
                        seen_targets.add(new_col)
                    else:
                        logger.debug(f"Skipping duplicate mapping: {old_col} -> {new_col}")

                column_mapping = filtered_mapping

            migrated_df = migrated_df.rename(columns=column_mapping)
            result.migrated_columns = column_mapping
            logger.info(f"Migrated {len(column_mapping)} columns to new schema")

        # Validate the migration
        if schema_type == "snapshot":
            # Check that we have the essential columns
            essential_columns = [SnapshotColumns.EMP_ID, SnapshotColumns.EMP_HIRE_DATE]
            missing_essential = [col for col in essential_columns if col not in migrated_df.columns]
            if missing_essential:
                result.warnings.append(
                    f"Missing essential columns after migration: {missing_essential}"
                )

        logger.info(
            f"Successfully migrated {schema_type} schema with {len(column_mapping)} column mappings"
        )

    except Exception as e:
        result.errors.append(f"Migration failed: {str(e)}")
        result.success = False
        return df, result

    return migrated_df, result


def create_compatibility_wrapper(df: pd.DataFrame, target_schema: str = "legacy") -> pd.DataFrame:
    """Create a compatibility wrapper that maps columns back to legacy names.

    Args:
        df: DataFrame with new schema column names
        target_schema: Target schema ("legacy" or "new")

    Returns:
        DataFrame with target schema column names
    """
    if target_schema == "legacy":
        # Map new names back to legacy names
        reverse_mapping = LegacyColumnMapper.get_reverse_mapping("snapshot")
        mapping = {
            new_name: old_name
            for new_name, old_name in reverse_mapping.items()
            if new_name in df.columns
        }
        return df.rename(columns=mapping)
    elif target_schema == "new":
        # No change needed if already in new schema
        return df
    else:
        raise ValueError(f"Unknown target schema: {target_schema}")


def detect_schema_version(df: pd.DataFrame) -> str:
    """Detect which schema version a DataFrame is using.

    Args:
        df: DataFrame to analyze

    Returns:
        Schema version ("legacy_emp_prefix", "new", or "unknown")
    """
    columns = set(df.columns)

    # Check for legacy EMP_ prefix schema first (more specific)
    legacy_emp_indicators = {"EMP_ID", "EMP_HIRE_DATE", "EMP_GROSS_COMP"}
    if legacy_emp_indicators.issubset(columns):
        return "legacy_emp_prefix"

    # Check for new schema (descriptive names)
    new_schema_core = {
        SnapshotColumns.EMP_ID.value,
        SnapshotColumns.EMP_HIRE_DATE.value,
        SnapshotColumns.EMP_GROSS_COMP.value,
    }

    # New schema should have these core columns
    if new_schema_core.issubset(columns):
        # Additional check: new schema typically has more structured columns
        new_schema_indicators = [
            SnapshotColumns.SIMULATION_YEAR.value,
            SnapshotColumns.EMP_ACTIVE.value,
            SnapshotColumns.IS_ELIGIBLE.value,
        ]

        # If it has several new schema indicators, it's the new schema
        new_indicators_count = sum(1 for col in new_schema_indicators if col in columns)
        if new_indicators_count >= 1:  # At least one new-style column
            return "new"
        else:
            # Has core columns but no new indicators - could be legacy or minimal new
            return "new"  # Default to new since we're using descriptive names

    return "unknown"


def get_migration_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """Get recommendations for migrating a DataFrame to the new schema.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with migration recommendations
    """
    schema_version = detect_schema_version(df)
    recommendations = {
        "current_schema": schema_version,
        "migration_needed": schema_version != "new",
        "recommendations": [],
        "potential_mappings": {},
        "confidence": "high",
    }

    if schema_version == "legacy_emp_prefix":
        recommendations["recommendations"].append(
            "Direct mapping available for EMP_ prefixed columns"
        )
        recommendations["potential_mappings"] = LegacyColumnMapper.get_snapshot_mapping(
            df.columns.tolist()
        )
        recommendations["confidence"] = "high"

    elif schema_version == "legacy_full_names":
        recommendations["recommendations"].append("Direct mapping available for full name columns")
        recommendations["potential_mappings"] = LegacyColumnMapper.get_snapshot_mapping(
            df.columns.tolist()
        )
        recommendations["confidence"] = "high"

    elif schema_version == "unknown":
        recommendations["recommendations"].append(
            "Schema not recognized, fuzzy matching recommended"
        )
        recommendations["potential_mappings"] = LegacyColumnMapper.fuzzy_match_columns(
            df.columns.tolist()
        )
        recommendations["confidence"] = "low"

        # Provide specific recommendations
        unmapped_cols = set(df.columns) - set(recommendations["potential_mappings"].keys())
        if unmapped_cols:
            recommendations["recommendations"].append(
                f"Manual review needed for columns: {unmapped_cols}"
            )

    elif schema_version == "new":
        recommendations["recommendations"].append("Already using new schema, no migration needed")

    return recommendations
