import os
import json
from typing import Dict, List, Optional, Any
from ftlite.feature import Entity, Feature, FeatureView, OnDemandFeatureView

class Registry:
    """Tracks defined features, entities, feature views, and on-demand feature views, persisting metadata to a JSON file."""
    def __init__(self, registry_path: str = ".ftlite/registry.json"):
        self.registry_path = registry_path
        self._entities: Dict[str, Entity] = {}
        self._feature_views: Dict[str, FeatureView] = {}
        self._on_demand_feature_views: Dict[str, OnDemandFeatureView] = {}
        self.load()

    def register_entity(self, entity: Entity) -> None:
        self._entities[entity.name] = entity
        self.save()

    def register_feature_view(self, feature_view: FeatureView) -> None:
        # Check that all entities in the feature view are registered
        for entity in feature_view.entities:
            if entity.name not in self._entities:
                self.register_entity(entity)
        self._feature_views[feature_view.name] = feature_view
        self.save()

    def register_on_demand_feature_view(self, ondemand_fv: OnDemandFeatureView) -> None:
        self._on_demand_feature_views[ondemand_fv.name] = ondemand_fv
        self.save()

    def get_entity(self, name: str) -> Entity:
        if name not in self._entities:
            raise KeyError(f"Entity '{name}' is not registered.")
        return self._entities[name]

    def get_feature_view(self, name: str) -> FeatureView:
        if name not in self._feature_views:
            raise KeyError(f"FeatureView '{name}' is not registered.")
        return self._feature_views[name]

    def get_on_demand_feature_view(self, name: str) -> OnDemandFeatureView:
        if name not in self._on_demand_feature_views:
            raise KeyError(f"OnDemandFeatureView '{name}' is not registered.")
        return self._on_demand_feature_views[name]

    def list_entities(self) -> List[Entity]:
        return list(self._entities.values())

    def list_feature_views(self) -> List[FeatureView]:
        return list(self._feature_views.values())

    def list_on_demand_feature_views(self) -> List[OnDemandFeatureView]:
        return list(self._on_demand_feature_views.values())

    def save(self) -> None:
        """Serializes current registry state to a JSON file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        data = {
            "entities": {
                name: {
                    "name": ent.name,
                    "value_type": ent.value_type,
                    "join_key": ent.join_key,
                    "description": ent.description
                }
                for name, ent in self._entities.items()
            },
            "feature_views": {
                name: {
                    "name": fv.name,
                    "entities": [ent.name for ent in fv.entities],
                    "features": [
                        {
                            "name": feat.name,
                            "dtype": feat.dtype,
                            "description": feat.description
                        }
                        for feat in fv.features
                    ],
                    "source_path": fv.source_path,
                    "timestamp_field": fv.timestamp_field,
                    "created_timestamp_field": fv.created_timestamp_field
                }
                for name, fv in self._feature_views.items()
            },
            "on_demand_feature_views": {
                name: {
                    "name": odfv.name,
                    "features": [
                        {
                            "name": feat.name,
                            "dtype": feat.dtype,
                            "description": feat.description
                        }
                        for feat in odfv.features
                    ],
                    "inputs": odfv.inputs
                }
                for name, odfv in self._on_demand_feature_views.items()
            }
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Deserializes registry state from JSON if the file exists."""
        if not os.path.exists(self.registry_path):
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            
            # Reconstruct entities
            for name, ent_data in data.get("entities", {}).items():
                self._entities[name] = Entity(
                    name=ent_data["name"],
                    value_type=ent_data["value_type"],
                    join_key=ent_data.get("join_key"),
                    description=ent_data.get("description")
                )
            
            # Reconstruct feature views
            for name, fv_data in data.get("feature_views", {}).items():
                entities = [self.get_entity(ent_name) for ent_name in fv_data["entities"]]
                features = [
                    Feature(
                        name=feat["name"],
                        dtype=feat["dtype"],
                        description=feat.get("description")
                    )
                    for feat in fv_data["features"]
                ]
                self._feature_views[name] = FeatureView(
                    name=fv_data["name"],
                    entities=entities,
                    features=features,
                    source_path=fv_data["source_path"],
                    timestamp_field=fv_data["timestamp_field"],
                    created_timestamp_field=fv_data.get("created_timestamp_field")
                )
                
            # Reconstruct on-demand feature views (without transform_fn since it is a function)
            for name, odfv_data in data.get("on_demand_feature_views", {}).items():
                features = [
                    Feature(
                        name=feat["name"],
                        dtype=feat["dtype"],
                        description=feat.get("description")
                    )
                    for feat in odfv_data["features"]
                ]
                # Default to None for transform_fn, to be registered by code at runtime
                self._on_demand_feature_views[name] = OnDemandFeatureView(
                    name=odfv_data["name"],
                    features=features,
                    inputs=odfv_data["inputs"],
                    transform_fn=None
                )
        except Exception as e:
            # If registry file is corrupted, start fresh or warn (here we initialize clean)
            print(f"Warning: Failed to load registry: {e}. Starting with an empty registry.")
            self._entities = {}
            self._feature_views = {}
            self._on_demand_feature_views = {}
