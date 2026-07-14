import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient
from generate_data import generate_churn_dataset


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Generate churn dataset if not present
    profile_path = os.path.join(current_dir, "customer_profiles.parquet")
    activity_path = os.path.join(current_dir, "customer_activities.parquet")
    labels_path = os.path.join(current_dir, "churn_labels.parquet")

    if not (
        os.path.exists(profile_path)
        and os.path.exists(activity_path)
        and os.path.exists(labels_path)
    ):
        print("Generating synthetic churn dataset...")
        generate_churn_dataset(current_dir)
        print("Dataset generated successfully.\n")

    # 2. Setup Ftlite Client
    registry_path = os.path.join(current_dir, ".ftlite", "registry.json")
    online_db_path = os.path.join(current_dir, ".ftlite", "online_store.db")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 3. Define and Register Entity
    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)
    print("Registered Customer Entity.")

    # 4. Define and Register Feature Views
    # Feature view for dynamic customer activities
    activity_fv = FeatureView(
        name="customer_activity",
        entities=[customer],
        features=[
            Feature(name="usage_minutes", dtype="double"),
            Feature(name="support_calls", dtype="int64"),
            Feature(name="active_days_in_week", dtype="int64"),
        ],
        source_path=activity_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(activity_fv)
    print("Registered customer_activity FeatureView.")

    # 5. Point-In-Time Join for Training Data Generation
    # Load training targets (the entity keys and observation timestamps)
    training_entities = pd.read_parquet(labels_path)
    print(f"\nLoaded {len(training_entities)} training targets (churn labels).")
    print("Sample observation targets:")
    print(training_entities.head(3))

    # Retrieve historical features for training
    features_to_fetch = ["usage_minutes", "support_calls", "active_days_in_week"]
    print(f"\nRunning point-in-time correct join for features: {features_to_fetch}...")

    training_data = client.get_historical_features(
        entity_df=training_entities,
        features=features_to_fetch,
        timestamp_col="timestamp",
    )

    print("\nJoined training data (Point-in-Time Correct):")
    print(training_data.head(5))

    # 6. Materialize Features to SQLite Online Store for Inference
    # Materialize features from June 1 to July 1, 2026
    start_time = datetime.datetime(2026, 6, 1)
    end_time = datetime.datetime(2026, 7, 1)
    print(
        f"\nMaterializing features to Online Store between {start_time} and {end_time}..."
    )
    client.materialize(start_time, end_time)
    print("Materialization complete.")

    # 7. Get Online Features for Low-Latency Prediction
    # Let's say we want to serve features in real-time for customer 10001, 10002, 10003
    inference_customers = [10001, 10002, 10003]
    print(f"\nRetrieving online features for customers: {inference_customers}...")
    online_features = client.get_online_features(
        entity_keys=inference_customers,
        features=["usage_minutes", "support_calls", "active_days_in_week"],
    )

    for f in online_features:
        print(f"Customer {f['entity_id']}: {f}")


if __name__ == "__main__":
    main()
