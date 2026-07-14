import pandas as pd
import duckdb
from typing import List
from ftlite.feature import FeatureView


class OfflineStore:
    """Manages historical feature data using DuckDB/Parquet and performs point-in-time correct joins."""

    def __init__(self, connection_string: str = ":memory:"):
        self.con = duckdb.connect(connection_string)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_views: List[FeatureView],
        feature_names: List[str],
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Retrieves historical features for a given entity dataframe with point-in-time correctness.
        Uses DuckDB's ASOF JOIN to match the entity timestamp with the latest feature value before or at that timestamp.
        """
        if entity_df.empty:
            return entity_df.copy()

        # Register the entity dataframe with DuckDB
        self.con.register("entity_df", entity_df)

        # We start with the base entity_df and join features from each feature view sequentially
        current_table = "entity_df"

        for fv in feature_views:
            # Find which requested features belong to this feature view
            fv_feat_names = [f.name for f in fv.features]
            requested_fv_feats = [f for f in feature_names if f in fv_feat_names]

            if not requested_fv_feats:
                continue

            # NOTE: SQL interpolation is used here for internal database identifiers.
            # This is not intended for untrusted user inputs (risk of SQL injection).
            source_query = f"SELECT * FROM read_parquet('{fv.source_path}')"
            self.con.register(f"source_{fv.name}", self.con.query(source_query).df())

            # Find the join key (usually the entity's join_key)
            # We assume a single entity join key for simplicity in the skeleton
            join_key = fv.entities[0].join_key

            # Build ASOF JOIN query:
            # DuckDB ASOF JOIN requires: table_a ASOF JOIN table_b ON a.key = b.key AND a.timestamp >= b.timestamp
            select_feats = ", ".join([f"b.{f}" for f in requested_fv_feats])

            # NOTE: Column/table names are interpolated here. Use only with trusted schema definitions.
            query = f"""
                SELECT 
                    a.*,
                    {select_feats}
                FROM {current_table} a
                ASOF LEFT JOIN source_{fv.name} b
                ON a.{join_key} = b.{join_key}
                AND a.{timestamp_col} >= b.{fv.timestamp_field}
            """

            # Run query and update current table in DuckDB
            result_df = self.con.query(query).df()
            current_table = f"temp_joined_{fv.name}"
            self.con.register(current_table, result_df)

        final_df = self.con.query(f"SELECT * FROM {current_table}").df()

        # Clean up temporary tables
        for fv in feature_views:
            try:
                self.con.unregister(f"source_{fv.name}")
            except Exception:
                pass
            try:
                self.con.unregister(f"temp_joined_{fv.name}")
            except Exception:
                pass
        try:
            self.con.unregister("entity_df")
        except Exception:
            pass

        return final_df
