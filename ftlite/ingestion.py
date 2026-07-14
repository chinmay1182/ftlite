import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ftlite.feature import FeatureView


def push_features(feature_view: FeatureView, df: pd.DataFrame) -> None:
    """
    Appends new feature records to the feature view's offline Parquet source path.
    Creates the directory and file if they do not exist.
    """
    if df.empty:
        return

    # Check if directory exists
    source_path = feature_view.source_path
    os.makedirs(os.path.dirname(os.path.abspath(source_path)), exist_ok=True)

    # Convert dataframe to PyArrow Table
    table = pa.Table.from_pandas(df)

    # Write or append to parquet
    if os.path.exists(source_path):
        # We can append by reading existing or writing partition/multiple files.
        # For simplicity, we can read existing, concatenate, and rewrite,
        # or write to a new parquet file if partitioned.
        # Let's read existing, concat and write back for a local file-based offline store.
        try:
            existing_table = pq.read_table(source_path)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, source_path)
        except Exception:
            # If read fails, just write the new table
            pq.write_table(table, source_path)
    else:
        pq.write_table(table, source_path)
