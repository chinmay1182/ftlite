import datetime
from typing import List, Any, Dict, Optional
import pandas as pd
import duckdb

from ftlite.feature import Entity, FeatureView, OnDemandFeatureView
from ftlite.registry import Registry
from ftlite.offline_store import OfflineStore
from ftlite.online_store import OnlineStore
from ftlite.ingestion import push_features

class FtliteClient:
    """The main client interface for Ftlite feature store operations."""
    def __init__(
        self,
        registry_path: str = ".ftlite/registry.json",
        online_db_path: str = ".ftlite/online_store.db",
        offline_connection_string: str = ":memory:"
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

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Retrieves point-in-time correct historical features (including on-demand transformations).
        
        Args:
            entity_df: DataFrame containing entity keys and timestamp column.
            features: List of feature names to retrieve (e.g. ['fv_name:feature_name']).
            timestamp_col: Name of the timestamp column in entity_df.
        """
        if entity_df.empty:
            return entity_df.copy()

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
            
            fv_name = None
            base_feat = feat_name
            if ":" in feat_name:
                fv_name, base_feat = feat_name.split(":", 1)
                
            is_on_demand = False
            for odfv in self.registry.list_on_demand_feature_views():
                if fv_name and odfv.name != fv_name:
                    continue
                if any(f.name == base_feat for f in odfv.features):
                    is_on_demand = True
                    if odfv not in on_demand_fvs_to_evaluate:
                        # Append to evaluation list
                        on_demand_fvs_to_evaluate.append(odfv)
                        # Add inputs to resolution queue
                        features_to_resolve.extend(odfv.inputs)
                    break
            
            if not is_on_demand:
                standard_features_to_fetch.add(feat_name)

        # 2. Fetch standard features from offline store
        result_df = entity_df.copy()
        if standard_features_to_fetch:
            fvs_to_query = []
            feature_names_to_fetch = []
            
            for feat_name in standard_features_to_fetch:
                fv_name = None
                base_feat = feat_name
                if ":" in feat_name:
                    fv_name, base_feat = feat_name.split(":", 1)
                    
                found = False
                for fv in self.registry.list_feature_views():
                    if fv_name and fv.name != fv_name:
                        continue
                    if any(f.name == base_feat for f in fv.features):
                        if fv not in fvs_to_query:
                            fvs_to_query.append(fv)
                        feature_names_to_fetch.append(base_feat)
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Feature '{feat_name}' not found in any registered FeatureView.")
            
            result_df = self.offline_store.get_historical_features(
                entity_df=entity_df,
                feature_views=fvs_to_query,
                feature_names=feature_names_to_fetch,
                timestamp_col=timestamp_col
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
                    if inp not in result_df.columns and inp_base not in result_df.columns:
                        inputs_present = False
                        break
                
                if inputs_present:
                    # Look up active transform_fn (it might be in memory, not from registry JSON load)
                    active_odfv = self.registry.get_on_demand_feature_view(odfv.name)
                    if active_odfv.transform_fn is None:
                        raise ValueError(f"OnDemandFeatureView '{odfv.name}' transform_fn is not registered at runtime.")
                    
                    transformed_df = active_odfv.transform_fn(result_df)
                    
                    for feat in odfv.features:
                        col_in_transformed = None
                        if feat.name in transformed_df.columns:
                            col_in_transformed = feat.name
                        elif f"{odfv.name}:{feat.name}" in transformed_df.columns:
                            col_in_transformed = f"{odfv.name}:{feat.name}"
                        
                        if col_in_transformed is not None:
                            result_df[feat.name] = transformed_df[col_in_transformed].values
                        else:
                            if feat.name in result_df.columns:
                                pass
                            else:
                                raise ValueError(f"Transformation function for '{odfv.name}' did not produce feature '{feat.name}'.")
                    
                    remaining_fvs.remove(odfv)
                    evaluated_any = True
                    
        if remaining_fvs:
            raise ValueError(f"Circular dependency or missing inputs for on-demand feature views: {[f.name for f in remaining_fvs]}")

        # 4. Map back requested output columns
        final_cols = list(entity_df.columns)
        for req_feat in features:
            base_feat = req_feat.split(":", 1)[1] if ":" in req_feat else req_feat
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
            base_feat = req_feat.split(":", 1)[1] if ":" in req_feat else req_feat
            
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

        return result_df[final_cols]

    def push(self, feature_view_name: str, df: pd.DataFrame) -> None:
        """Pushes/appends raw feature records into the offline Parquet file."""
        fv = self.registry.get_feature_view(feature_view_name)
        push_features(fv, df)

    def materialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        feature_views: Optional[List[str]] = None
    ) -> None:
        """
        Materializes/syncs features from the offline store to the online store for a time window.
        """
        fvs_to_materialize = []
        if feature_views:
            for name in feature_views:
                fvs_to_materialize.append(self.registry.get_feature_view(name))
        else:
            fvs_to_materialize = self.registry.list_feature_views()

        con = duckdb.connect(":memory:")
        for fv in fvs_to_materialize:
            con.execute(f"CREATE OR REPLACE TEMPORARY TABLE temp_fv AS SELECT * FROM read_parquet('{fv.source_path}')")
            
            query = f"""
                SELECT * FROM temp_fv
                WHERE {fv.timestamp_field} >= ? AND {fv.timestamp_field} <= ?
            """
            
            df = con.execute(query, (start_time.isoformat(), end_time.isoformat())).df()
            self.online_store.write_features(fv, df)

    def get_online_features(
        self,
        entity_keys: List[Any],
        features: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieves low-latency online features (including on-demand transformations).
        """
        standard_features_to_fetch = set()
        on_demand_fvs_to_evaluate = []
        
        features_to_resolve = list(features)
        resolved_features = set()
        
        while features_to_resolve:
            feat_name = features_to_resolve.pop(0)
            if feat_name in resolved_features:
                continue
            resolved_features.add(feat_name)
            
            fv_name = None
            base_feat = feat_name
            if ":" in feat_name:
                fv_name, base_feat = feat_name.split(":", 1)
                
            is_on_demand = False
            for odfv in self.registry.list_on_demand_feature_views():
                if fv_name and odfv.name != fv_name:
                    continue
                if any(f.name == base_feat for f in odfv.features):
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
                fv_name = None
                base_feat = feat_name
                if ":" in feat_name:
                    fv_name, base_feat = feat_name.split(":", 1)
                    
                found = False
                for fv in self.registry.list_feature_views():
                    if fv_name and fv.name != fv_name:
                        continue
                    if any(f.name == base_feat for f in fv.features):
                        if fv not in fvs_to_query:
                            fvs_to_query.append(fv)
                        feature_names_to_fetch.append(base_feat)
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Feature '{feat_name}' not found in any registered FeatureView.")
            
            online_results = self.online_store.get_online_features(
                entity_keys=entity_keys,
                feature_names=feature_names_to_fetch,
                feature_views=fvs_to_query
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
                        raise ValueError(f"OnDemandFeatureView '{odfv.name}' transform_fn is not registered at runtime.")
                    
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
                                raise ValueError(f"Transformation function for '{odfv.name}' did not produce feature '{feat.name}'.")
                    
                    remaining_fvs.remove(odfv)
                    evaluated_any = True
                    
        if remaining_fvs:
            raise ValueError(f"Circular dependency or missing inputs for on-demand feature views: {[f.name for f in remaining_fvs]}")

        # 4. Map back to List[Dict[str, Any]]
        final_results = []
        for idx, key in enumerate(entity_keys):
            row = df.iloc[idx]
            res = {"entity_id": key}
            for req_feat in features:
                base_feat = req_feat.split(":", 1)[1] if ":" in req_feat else req_feat
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
