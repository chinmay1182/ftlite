import datetime
import os
import hashlib
import pickle
import time
import pandas as pd
import duckdb
from typing import List, Any, Dict, Optional, Union, Tuple

from ftlite.feature import Entity, FeatureView, OnDemandFeatureView
from ftlite.registry import Registry
from ftlite.offline_store import OfflineStore
from ftlite.online_store import OnlineStore
from ftlite.ingestion import push_features

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def parse_feature_uri(
    feature_uri: str,
) -> Tuple[Optional[str], Optional[str], str]:
    """Parses a feature URI string into fv_name, version, and feature_name.

    Handles fv_name:feature_name, fv_name@version:feature_name, and
    fv_name:feature_name@version format.
    """
    fv_part = None
    feat_part = feature_uri
    if ":" in feature_uri:
        fv_part, feat_part = feature_uri.split(":", 1)

    version = None
    if fv_part and "@" in fv_part:
        fv_part, version = fv_part.split("@", 1)
    elif "@" in feat_part:
        feat_part, version = feat_part.split("@", 1)

    return fv_part, version, feat_part


def clean_cache(cache_dir: str, ttl_seconds: int = 86400, max_files: int = 100) -> None:
    """Cleans cache files that have expired or exceed the maximum allowed file count."""
    if not os.path.exists(cache_dir):
        return
    now = time.time()
    files = []
    for f in os.listdir(cache_dir):
        path = os.path.join(cache_dir, f)
        if not f.endswith(".parquet"):
            continue
        mtime = os.path.getmtime(path)
        if now - mtime > ttl_seconds:
            try:
                os.remove(path)
            except Exception:
                pass
        else:
            files.append((mtime, path))

    if len(files) > max_files:
        files.sort(key=lambda x: x[0])
        excess = len(files) - max_files
        for i in range(excess):
            try:
                os.remove(files[i][1])
            except Exception:
                pass


class FtliteClient:
    """The main client interface for Ftlite feature store operations."""

    def __init__(
        self,
        registry_path: str = ".ftlite/registry.json",
        online_db_path: str = ".ftlite/online_store.db",
        offline_connection_string: str = ":memory:",
    ):
        self.registry = Registry(registry_path)
        self.online_store = OnlineStore(online_db_path)
        self.offline_store = OfflineStore(offline_connection_string)

    def register_entity(self, entity: Entity) -> None:
        """Registers a new Entity in the feature store."""
        self.registry.register_entity(entity)

    def register_feature_view(self, feature_view: FeatureView) -> None:
        """Registers a new FeatureView in the feature store."""
        self.registry.register_feature_view(feature_view)

    def register_on_demand_feature_view(self, ondemand_fv: OnDemandFeatureView) -> None:
        """Registers a new OnDemandFeatureView in the feature store."""
        self.registry.register_on_demand_feature_view(ondemand_fv)

    def resolve_feature_view(
        self, fv_part: Optional[str], version: Optional[str], base_feat: str
    ) -> Optional[FeatureView]:
        """Resolves the correct FeatureView matching a feature name and optional version."""
        if fv_part:
            lookup_name = f"{fv_part}@{version}" if version else fv_part
            try:
                return self.registry.get_feature_view(lookup_name)
            except ValueError:
                return None

        # Find matching standard feature views
        matching_fvs = []
        for fv in self.registry.list_feature_views():
            if any(f.name == base_feat for f in fv.features):
                if version and fv.version != version:
                    continue
                matching_fvs.append(fv)

        if not matching_fvs:
            return None

        # Fallback to the latest version matching
        matching_fvs.sort(key=lambda x: x.version or "", reverse=True)
        return matching_fvs[0]

    def clear_cache(self, cache_dir: str = ".ftlite/cache") -> None:
        """Clears all cached Parquet files from the local cache directory."""
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith(".parquet"):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except Exception:
                        pass

    def get_historical_features(
        self,
        entity_df: Union[pd.DataFrame, Any],
        features: List[str],
        timestamp_col: str = "timestamp",
        output_format: str = "pandas",
        cache: bool = False,
        cache_dir: str = ".ftlite/cache",
        cache_ttl: int = 86400,
        cache_max_files: int = 100,
    ) -> Union[pd.DataFrame, Any]:
        """Retrieves point-in-time correct historical features (including on-demand transformations).

        Args:
            entity_df: DataFrame containing entity keys and timestamp column (Pandas or Polars).
            features: List of feature names to retrieve (e.g. ['fv_name:feature_name']).
            timestamp_col: Name of the timestamp column in entity_df.
            output_format: Format of returned DataFrame ('pandas' or 'polars').
            cache: Enable/disable caching of final results.
            cache_dir: Path to storage directory for caches.
            cache_ttl: Time-to-live in seconds for cache files.
            cache_max_files: Maximum number of cache files allowed on disk.
        """
        # Handle Polars vs Pandas input
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

        # Cache check
        cache_path = None
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            clean_cache(cache_dir, ttl_seconds=cache_ttl, max_files=cache_max_files)
            hasher = hashlib.sha256()
            hasher.update(pickle.dumps(sorted(features)))
            hasher.update(timestamp_col.encode())
            hasher.update(pickle.dumps(entity_df_pd.to_dict(orient="list")))
            cache_key = hasher.hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")
            if os.path.exists(cache_path):
                if output_format == "polars":
                    if not HAS_POLARS:
                        raise ImportError(
                            "polars is not installed. Please install it using 'pip install ftlite[polars]' to return Polars DataFrames."
                        )
                    return pl.read_parquet(cache_path)
                return pd.read_parquet(cache_path)

        # 1. Resolve requested features into standard and on-demand
        standard_features_to_fetch = set()
        on_demand_fvs_to_evaluate = []

        features_to_resolve = list(features)
        resolved_features = set()

        while features_to_resolve:
            feat_name = features_to_resolve.pop(0)
            if feat_name in resolved_features:
                continue
            resolved_features.add(feat_name)

            fv_part, version, base_feat = parse_feature_uri(feat_name)

            is_on_demand = False
            for odfv in self.registry.list_on_demand_feature_views():
                if fv_part and odfv.base_name != fv_part and odfv.name != fv_part:
                    continue
                if any(f.name == base_feat for f in odfv.features):
                    if version and odfv.version != version:
                        continue
                    is_on_demand = True
                    if odfv not in on_demand_fvs_to_evaluate:
                        on_demand_fvs_to_evaluate.append(odfv)
                        features_to_resolve.extend(odfv.inputs)
                    break

            if not is_on_demand:
                standard_features_to_fetch.add(feat_name)

        # 2. Fetch standard features from offline store
        result_df = entity_df_pd.copy()
        if standard_features_to_fetch:
            fvs_to_query = []
            feature_names_to_fetch = []

            for feat_name in standard_features_to_fetch:
                fv_part, version, base_feat = parse_feature_uri(feat_name)

                # Resolve through custom helper with version fallback lookup
                found_fv = self.resolve_feature_view(fv_part, version, base_feat)

                if found_fv and any(f.name == base_feat for f in found_fv.features):
                    if found_fv not in fvs_to_query:
                        fvs_to_query.append(found_fv)
                    feature_names_to_fetch.append(base_feat)
                else:
                    raise ValueError(
                        f"Feature '{feat_name}' not found in any registered FeatureView."
                    )

            result_df = self.offline_store.get_historical_features(
                entity_df=entity_df_pd,
                feature_views=fvs_to_query,
                feature_names=feature_names_to_fetch,
                timestamp_col=timestamp_col,
            )

        # 3. Evaluate On-Demand Feature Views
        remaining_fvs = list(on_demand_fvs_to_evaluate)
        evaluated_any = True
        while remaining_fvs and evaluated_any:
            evaluated_any = False
            for odfv in list(remaining_fvs):
                inputs_present = True
                for inp in odfv.inputs:
                    inp_base = inp.split(":", 1)[1] if ":" in inp else inp
                    if (
                        inp not in result_df.columns
                        and inp_base not in result_df.columns
                    ):
                        inputs_present = False
                        break

                if inputs_present:
                    active_odfv = self.registry.get_on_demand_feature_view(odfv.name)
                    if active_odfv.transform_fn is None:
                        raise ValueError(
                            f"OnDemandFeatureView '{odfv.name}' transform_fn is not registered at runtime."
                        )

                    transformed_df = active_odfv.transform_fn(result_df)

                    for feat in odfv.features:
                        col_in_transformed = None
                        if feat.name in transformed_df.columns:
                            col_in_transformed = feat.name
                        elif f"{odfv.name}:{feat.name}" in transformed_df.columns:
                            col_in_transformed = f"{odfv.name}:{feat.name}"

                        if col_in_transformed is not None:
                            result_df[feat.name] = transformed_df[
                                col_in_transformed
                            ].values
                        else:
                            if feat.name in result_df.columns:
                                pass
                            else:
                                raise ValueError(
                                    f"Transformation function for '{odfv.name}' did not produce feature '{feat.name}'."
                                )

                    remaining_fvs.remove(odfv)
                    evaluated_any = True

        if remaining_fvs:
            raise ValueError(
                f"Circular dependency or missing inputs for on-demand feature views: {[f.name for f in remaining_fvs]}"
            )

        # 4. Map back requested output columns
        final_cols = list(entity_df_pd.columns)
        for req_feat in features:
            fv_part, version, base_feat = parse_feature_uri(req_feat)
            if req_feat in result_df.columns:
                final_cols.append(req_feat)
            elif base_feat in result_df.columns:
                if req_feat != base_feat:
                    result_df[req_feat] = result_df[base_feat]
                final_cols.append(req_feat)

        # 5. Type casting based on registry schema (e.g. nullable Int64 for integers)
        for req_feat in features:
            if req_feat not in result_df.columns:
                continue
            fv_part, version, base_feat = parse_feature_uri(req_feat)

            # Find feature dtype
            feature_dtype = None
            for fv in self.registry.list_feature_views():
                for f in fv.features:
                    if f.name == base_feat:
                        feature_dtype = f.dtype
                        break
                if feature_dtype:
                    break
            if not feature_dtype:
                for odfv in self.registry.list_on_demand_feature_views():
                    for f in odfv.features:
                        if f.name == base_feat:
                            feature_dtype = f.dtype
                            break
                    if feature_dtype:
                        break

            if feature_dtype:
                dtype_lower = feature_dtype.lower()
                if "int" in dtype_lower:
                    result_df[req_feat] = result_df[req_feat].astype("Int64")
                elif "float" in dtype_lower or "double" in dtype_lower:
                    result_df[req_feat] = result_df[req_feat].astype("float64")
                elif "bool" in dtype_lower:
                    result_df[req_feat] = result_df[req_feat].astype("boolean")
                elif "string" in dtype_lower or "text" in dtype_lower:
                    result_df[req_feat] = result_df[req_feat].astype("string")

        output_df = result_df[final_cols]

        # Write to cache if enabled
        if cache and cache_path:
            output_df.to_parquet(cache_path)

        if output_format == "polars":
            if not HAS_POLARS:
                raise ImportError(
                    "polars is not installed. Please install it using 'pip install ftlite[polars]' to return Polars DataFrames."
                )
            return pl.DataFrame(output_df)
        return output_df

    def push(self, feature_view_name: str, df: pd.DataFrame) -> None:
        """Pushes/appends raw feature records into the offline Parquet file."""
        fv = self.registry.get_feature_view(feature_view_name)
        push_features(fv, df)

    def materialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        feature_views: Optional[List[str]] = None,
    ) -> None:
        """Materializes/syncs features from the offline store to the online store for a time window."""
        fvs_to_materialize = []
        if feature_views:
            for name in feature_views:
                fvs_to_materialize.append(self.registry.get_feature_view(name))
        else:
            fvs_to_materialize = self.registry.list_feature_views()

        con = duckdb.connect(":memory:")
        for fv in fvs_to_materialize:
            con.execute(
                f"CREATE OR REPLACE TEMPORARY TABLE temp_fv AS SELECT * FROM read_parquet('{fv.source_path}')"
            )

            query = f"""
                SELECT * FROM temp_fv
                WHERE {fv.timestamp_field} >= ? AND {fv.timestamp_field} <= ?
            """

            df = con.execute(query, (start_time.isoformat(), end_time.isoformat())).df()
            self.online_store.write_features(fv, df)

    def get_online_features(
        self, entity_keys: List[Any], features: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieves low-latency online features (including on-demand transformations)."""
        standard_features_to_fetch = set()
        on_demand_fvs_to_evaluate = []

        features_to_resolve = list(features)
        resolved_features = set()

        while features_to_resolve:
            feat_name = features_to_resolve.pop(0)
            if feat_name in resolved_features:
                continue
            resolved_features.add(feat_name)

            fv_part, version, base_feat = parse_feature_uri(feat_name)

            is_on_demand = False
            for odfv in self.registry.list_on_demand_feature_views():
                if fv_part and odfv.base_name != fv_part and odfv.name != fv_part:
                    continue
                if any(f.name == base_feat for f in odfv.features):
                    if version and odfv.version != version:
                        continue
                    is_on_demand = True
                    if odfv not in on_demand_fvs_to_evaluate:
                        on_demand_fvs_to_evaluate.append(odfv)
                        features_to_resolve.extend(odfv.inputs)
                    break

            if not is_on_demand:
                standard_features_to_fetch.add(feat_name)

        # 2. Fetch standard features from online store
        online_results = []
        if standard_features_to_fetch:
            fvs_to_query = []
            feature_names_to_fetch = []

            for feat_name in standard_features_to_fetch:
                fv_part, version, base_feat = parse_feature_uri(feat_name)

                # Resolve with version fallback
                found_fv = self.resolve_feature_view(fv_part, version, base_feat)

                if found_fv and any(f.name == base_feat for f in found_fv.features):
                    if found_fv not in fvs_to_query:
                        fvs_to_query.append(found_fv)
                    feature_names_to_fetch.append(base_feat)
                else:
                    raise ValueError(
                        f"Feature '{feat_name}' not found in any registered FeatureView."
                    )

            online_results = self.online_store.get_online_features(
                entity_keys=entity_keys,
                feature_names=feature_names_to_fetch,
                feature_views=fvs_to_query,
            )
        else:
            online_results = [{"entity_id": k} for k in entity_keys]

        # Convert to Pandas DataFrame for transformation function input
        join_key = "entity_id"
        if self.registry.list_entities():
            join_key = self.registry.list_entities()[0].join_key

        df_records = []
        for res in online_results:
            record = {join_key: res["entity_id"]}
            for k, v in res.items():
                if k != "entity_id":
                    record[k] = v
            df_records.append(record)

        df = pd.DataFrame(df_records)

        # 3. Evaluate On-Demand Feature Views
        remaining_fvs = list(on_demand_fvs_to_evaluate)
        evaluated_any = True
        while remaining_fvs and evaluated_any:
            evaluated_any = False
            for odfv in list(remaining_fvs):
                inputs_present = True
                for inp in odfv.inputs:
                    inp_base = inp.split(":", 1)[1] if ":" in inp else inp
                    if inp not in df.columns and inp_base not in df.columns:
                        inputs_present = False
                        break

                if inputs_present:
                    active_odfv = self.registry.get_on_demand_feature_view(odfv.name)
                    if active_odfv.transform_fn is None:
                        raise ValueError(
                            f"OnDemandFeatureView '{odfv.name}' transform_fn is not registered at runtime."
                        )

                    transformed_df = active_odfv.transform_fn(df)

                    for feat in odfv.features:
                        col_in_transformed = None
                        if feat.name in transformed_df.columns:
                            col_in_transformed = feat.name
                        elif f"{odfv.name}:{feat.name}" in transformed_df.columns:
                            col_in_transformed = f"{odfv.name}:{feat.name}"

                        if col_in_transformed is not None:
                            df[feat.name] = transformed_df[col_in_transformed].values
                        else:
                            if feat.name in df.columns:
                                pass
                            else:
                                raise ValueError(
                                    f"Transformation function for '{odfv.name}' did not produce feature '{feat.name}'."
                                )

                    remaining_fvs.remove(odfv)
                    evaluated_any = True

        if remaining_fvs:
            raise ValueError(
                f"Circular dependency or missing inputs for on-demand feature views: {[f.name for f in remaining_fvs]}"
            )

        # 4. Map back to List[Dict[str, Any]]
        final_results = []
        for idx, key in enumerate(entity_keys):
            row = df.iloc[idx]
            res = {"entity_id": key}
            for req_feat in features:
                fv_part, version, base_feat = parse_feature_uri(req_feat)
                val = None
                if req_feat in df.columns:
                    val = row[req_feat]
                elif base_feat in df.columns:
                    val = row[base_feat]

                # Find feature dtype
                feature_dtype = None
                for fv in self.registry.list_feature_views():
                    for f in fv.features:
                        if f.name == base_feat:
                            feature_dtype = f.dtype
                            break
                    if feature_dtype:
                        break
                if not feature_dtype:
                    for odfv in self.registry.list_on_demand_feature_views():
                        for f in odfv.features:
                            if f.name == base_feat:
                                feature_dtype = f.dtype
                                break
                        if feature_dtype:
                            break

                # Convert pandas/numpy nan to None, and numpy values to native Python types
                if pd.isna(val):
                    val = None
                else:
                    if hasattr(val, "item"):
                        val = val.item()
                    if feature_dtype and "int" in feature_dtype.lower():
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            pass
                    elif feature_dtype and "bool" in feature_dtype.lower():
                        try:
                            val = bool(val)
                        except (ValueError, TypeError):
                            pass

                res[req_feat] = val
            final_results.append(res)

        return final_results

    def get_feature_lineage(self, feature_name: str) -> dict:
        """Traces the lineage of a given feature and returns dependency information."""
        fv_part, version, base_feat = parse_feature_uri(feature_name)

        # Check on-demand first
        for odfv in self.registry.list_on_demand_feature_views():
            if fv_part and odfv.name != fv_part and odfv.base_name != fv_part:
                continue
            for f in odfv.features:
                if f.name == base_feat:
                    # Found on-demand feature!
                    inputs_lineage = {}
                    for inp in odfv.inputs:
                        inputs_lineage[inp] = self.get_feature_lineage(inp)
                    return {
                        "name": f"{odfv.name}:{f.name}",
                        "type": "on_demand",
                        "dtype": f.dtype,
                        "feature_view": odfv.name,
                        "transform_fn": (
                            odfv.transform_fn.__name__
                            if odfv.transform_fn
                            else "registered_at_runtime"
                        ),
                        "inputs": inputs_lineage,
                    }

        # Check standard features
        for fv in self.registry.list_feature_views():
            if fv_part and fv.name != fv_part and fv.base_name != fv_part:
                continue
            for f in fv.features:
                if f.name == base_feat:
                    # Found standard feature!
                    return {
                        "name": f"{fv.name}:{f.name}",
                        "type": "standard",
                        "dtype": f.dtype,
                        "feature_view": fv.name,
                        "source_path": fv.source_path,
                        "entities": [e.name for e in fv.entities],
                    }

        raise ValueError(f"Feature '{feature_name}' not found in registry.")
