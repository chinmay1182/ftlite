import pandas as pd
import duckdb
from typing import List, Union, Any
from ftlite.feature import FeatureView

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def clean_db_name(name: str) -> str:
    """Sanitizes names containing '@' or ':' to be safe unquoted SQL identifiers."""
    return (
        name.replace("@", "_at_").replace(":", "_").replace("-", "_").replace(".", "_")
    )


class OfflineStore:
    """Manages historical feature data using DuckDB/Parquet and performs point-in-time correct joins."""

    def __init__(self, connection_string: str = ":memory:"):
        self.con = duckdb.connect(connection_string)

    def get_historical_features(
        self,
        entity_df: Union[pd.DataFrame, Any],
        feature_views: List[FeatureView],
        feature_names: List[str],
        timestamp_col: str = "timestamp",
        output_format: str = "pandas",
    ) -> Union[pd.DataFrame, Any]:
        """Retrieves historical features for a given entity dataframe with point-in-time correctness.

        Supports native Polars inputs/outputs (when polars is installed).
        """
        # Handle Polars vs Pandas DataFrame input
        if HAS_POLARS and isinstance(entity_df, pl.DataFrame):
            entity_df_pd = entity_df.to_pandas()
        elif type(
            entity_df
        ).__name__ == "DataFrame" and entity_df.__class__.__module__.startswith(
            "polars"
        ):
            raise ImportError(
                "polars is not installed. Please install it using 'pip install ftlite[polars]' to use Polars DataFrames."
            )
        else:
            entity_df_pd = entity_df

        if entity_df_pd.empty:
            if output_format == "polars":
                if not HAS_POLARS:
                    raise ImportError(
                        "polars is not installed. Please install it using 'pip install ftlite[polars]' to return Polars DataFrames."
                    )
                return pl.DataFrame(entity_df_pd)
            return entity_df_pd.copy()

        # Register the entity dataframe with DuckDB
        self.con.register("entity_df", entity_df_pd)

        # We start with the base entity_df and join features from each feature view sequentially
        current_table = "entity_df"

        for fv in feature_views:
            # Find which requested features belong to this feature view
            fv_feat_names = [f.name for f in fv.features]
            requested_fv_feats = [f for f in feature_names if f in fv_feat_names]

            if not requested_fv_feats:
                continue

            cleaned_fv_name = clean_db_name(fv.name)
            source_query = f"SELECT * FROM read_parquet('{fv.source_path}')"
            self.con.register(
                f"source_{cleaned_fv_name}", self.con.query(source_query).df()
            )

            # Find the join key
            join_key = fv.entities[0].join_key

            # Build ASOF JOIN query:
            select_feats = ", ".join([f"b.{f}" for f in requested_fv_feats])

            query = f"""
                SELECT 
                    a.*,
                    {select_feats}
                FROM {current_table} a
                ASOF LEFT JOIN source_{cleaned_fv_name} b
                ON a.{join_key} = b.{join_key}
                AND a.{timestamp_col} >= b.{fv.timestamp_field}
            """

            # Run query and update current table in DuckDB
            result_df = self.con.query(query).df()
            current_table = f"temp_joined_{cleaned_fv_name}"
            self.con.register(current_table, result_df)

        final_df = self.con.query(f"SELECT * FROM {current_table}").df()

        # Clean up temporary tables
        for fv in feature_views:
            cleaned_fv_name = clean_db_name(fv.name)
            try:
                self.con.unregister(f"source_{cleaned_fv_name}")
            except Exception:
                pass
            try:
                self.con.unregister(f"temp_joined_{cleaned_fv_name}")
            except Exception:
                pass
        try:
            self.con.unregister("entity_df")
        except Exception:
            pass

        if output_format == "polars":
            if not HAS_POLARS:
                raise ImportError(
                    "polars is not installed. Please install it using 'pip install ftlite[polars]' to return Polars DataFrames."
                )
            return pl.DataFrame(final_df)
        return final_df
