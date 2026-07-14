from typing import List, Optional, Callable


class Entity:
    """Represents a primary key or entity in the feature store."""

    def __init__(
        self,
        name: str,
        value_type: str,
        join_key: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.value_type = value_type  # e.g., "INT64", "STRING"
        self.join_key = join_key or name
        self.description = description

    def __repr__(self) -> str:
        return f"Entity(name={self.name}, join_key={self.join_key}, value_type={self.value_type})"


class Feature:
    """Represents an individual feature definition."""

    def __init__(self, name: str, dtype: str, description: Optional[str] = None):
        self.name = name
        self.dtype = dtype  # e.g., "FLOAT", "INT64", "STRING"
        self.description = description

    def __repr__(self) -> str:
        return f"Feature(name={self.name}, dtype={self.dtype})"


class FeatureView:
    """A logical grouping of related features, tied to one or more entities and a data source."""

    def __init__(
        self,
        name: str,
        entities: List[Entity],
        features: List[Feature],
        source_path: str,
        timestamp_field: str,
        created_timestamp_field: Optional[str] = None,
        version: Optional[str] = None,
    ):
        self.base_name = name
        self.version = version
        self.name = f"{name}@{version}" if version else name
        self.entities = entities
        self.features = features
        self.source_path = source_path
        self.timestamp_field = timestamp_field
        self.created_timestamp_field = created_timestamp_field

    def __repr__(self) -> str:
        feature_names = [f.name for f in self.features]
        entity_names = [e.name for e in self.entities]
        return f"FeatureView(name={self.name}, entities={entity_names}, features={feature_names})"


class OnDemandFeatureView:
    """Computes features on the fly using standard features or other inputs."""

    def __init__(
        self,
        name: str,
        features: List[Feature],
        inputs: List[str],
        transform_fn: Callable,
        version: Optional[str] = None,
    ):
        self.base_name = name
        self.version = version
        self.name = f"{name}@{version}" if version else name
        self.features = features
        self.inputs = inputs
        self.transform_fn = transform_fn

    def __repr__(self) -> str:
        feature_names = [f.name for f in self.features]
        return f"OnDemandFeatureView(name={self.name}, inputs={self.inputs}, features={feature_names})"
