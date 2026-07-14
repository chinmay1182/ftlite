import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any
from ftlite.feature import FeatureView

class OnlineStore:
    """Manages low-latency online feature retrieval using SQLite."""
    def __init__(self, db_path: str = ".ftlite/online_store.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        # Ensure parent directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with sqlite3.connect(self.db_path):
            pass

    def write_features(self, feature_view: FeatureView, df: pd.DataFrame) -> None:
        """
        Materializes/writes features from a DataFrame into the online SQLite database.
        Overwrites existing rows for the same entity key (upsert).
        """
        if df.empty:
            return

        join_key = feature_view.entities[0].join_key
        feature_names = [f.name for f in feature_view.features]
        timestamp_col = feature_view.timestamp_field
        
        # Columns to keep: join_key, features, timestamp
        cols_to_keep = [join_key] + feature_names + [timestamp_col]
        write_df = df[cols_to_keep].copy()
        
        # Keep only the latest record per entity (based on timestamp)
        write_df = write_df.sort_values(timestamp_col).groupby(join_key).last().reset_index()

        table_name = f"fv_{feature_view.name}"
        
        with sqlite3.connect(self.db_path) as conn:
            # We want to do an upsert or replace.
            # We can write to a temporary table and then insert or replace, or use SQLite's REPLACE INTO.
            # For simplicity, pandas to_sql can create/replace or we can write custom upsert logic.
            # Let's do a simple approach: read existing, merge/update, and write back, or use direct SQL INSERT OR REPLACE.
            
            # 1. Create table structure if it doesn't exist
            # We can use pandas to create the table structure if empty, or execute SQL.
            cursor = conn.cursor()
            
            # Build column definitions
            col_defs = [f"{join_key} TEXT PRIMARY KEY"]
            for feat in feature_view.features:
                # Basic mapping from python/parquet types to sqlite types
                sql_type = "REAL" if "float" in feat.dtype.lower() or "double" in feat.dtype.lower() else "TEXT"
                if "int" in feat.dtype.lower():
                    sql_type = "INTEGER"
                col_defs.append(f"{feat.name} {sql_type}")
            col_defs.append(f"{timestamp_col} TEXT")
            
            # NOTE: Table/column names are interpolated. Use only with trusted schema identifiers (SQL injection risk).
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})")
            
            # 2. Insert or Replace records
            cols = [join_key] + feature_names + [timestamp_col]
            placeholders = ", ".join(["?"] * len(cols))
            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"
            
            records = write_df[cols].values.tolist()
            cursor.executemany(insert_sql, records)
            conn.commit()

    def get_online_features(
        self,
        entity_keys: List[Any],
        feature_names: List[str],
        feature_views: List[FeatureView]
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the latest feature values for a list of entity keys.
        """
        results = []
        for entity_key in entity_keys:
            res = {"entity_id": entity_key}
            
            for fv in feature_views:
                join_key = fv.entities[0].join_key
                fv_feat_names = [f.name for f in fv.features]
                requested_fv_feats = [f for f in feature_names if f in fv_feat_names]
                
                if not requested_fv_feats:
                    continue
                
                table_name = f"fv_{fv.name}"
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # Check if table exists
                    # NOTE: Table name is interpolated. Use only with trusted identifiers.
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    if not cursor.fetchone():
                        # Table doesn't exist yet, populate None
                        for feat in requested_fv_feats:
                            res[feat] = None
                        continue
                        
                    # Query latest values for this entity_key
                    # NOTE: Column and table names are interpolated. Use only with trusted identifiers.
                    select_cols = ", ".join(requested_fv_feats)
                    query = f"SELECT {select_cols} FROM {table_name} WHERE {join_key} = ?"
                    cursor.execute(query, (str(entity_key),))
                    row = cursor.fetchone()
                    
                    if row:
                        for i, feat in enumerate(requested_fv_feats):
                            res[feat] = row[i]
                    else:
                        for feat in requested_fv_feats:
                            res[feat] = None
                            
            results.append(res)
            
        return results
