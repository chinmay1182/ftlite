import click
import datetime
import os
from ftlite.client import FtliteClient

@click.group()
def main():
    """FTLite command-line interface."""
    pass

@main.command()
@click.option("--registry", default=".ftlite/registry.json", help="Path to the registry JSON file.")
def init(registry):
    """Initializes the FTLite local directory and registry file."""
    db_dir = os.path.dirname(os.path.abspath(registry))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    if not os.path.exists(registry):
        import json
        with open(registry, "w") as f:
            json.dump({"entities": {}, "feature_views": {}, "on_demand_feature_views": {}}, f, indent=2)
        click.echo(f"Initialized empty registry at {registry}")
    else:
        click.echo(f"Registry already exists at {registry}")

@main.command()
@click.option("--registry", default=".ftlite/registry.json", help="Path to registry JSON file.")
@click.option("--online-db", default=".ftlite/online_store.db", help="Path to online store database file.")
@click.option("--start", required=True, help="Start time in ISO 8601 format (e.g. 2026-06-01T00:00:00).")
@click.option("--end", required=True, help="End time in ISO 8601 format (e.g. 2026-07-01T00:00:00).")
@click.option("--views", help="Comma-separated list of feature views to materialize. If omitted, all views are materialized.")
def materialize(registry, online_db, start, end, views):
    """Syncs features from offline parquet sources to SQLite online store for a time window."""
    try:
        start_dt = datetime.datetime.fromisoformat(start)
        end_dt = datetime.datetime.fromisoformat(end)
    except ValueError as e:
        click.echo(f"Error parsing dates: {e}. Please use ISO 8601 format.", err=True)
        return

    view_list = None
    if views:
        view_list = [v.strip() for v in views.split(",")]

    click.echo(f"Materializing features from {start_dt} to {end_dt}...")
    try:
        client = FtliteClient(registry_path=registry, online_db_path=online_db)
        client.materialize(start_dt, end_dt, feature_views=view_list)
        click.echo("Materialization complete.")
    except Exception as e:
        click.echo(f"Materialization failed: {e}", err=True)

@main.command()
@click.option("--registry", default=".ftlite/registry.json", help="Path to registry JSON file.")
def list(registry):
    """Lists registered Entities, FeatureViews, and OnDemandFeatureViews."""
    if not os.path.exists(registry):
        click.echo(f"Registry file not found at {registry}. Run 'ftlite init' first.", err=True)
        return

    client = FtliteClient(registry_path=registry)
    entities = client.registry.list_entities()
    feature_views = client.registry.list_feature_views()
    od_views = client.registry.list_on_demand_feature_views()

    click.echo("=== Entities ===")
    for ent in entities:
        click.echo(f"  {ent.name} (join_key: {ent.join_key}, value_type: {ent.value_type})")
        
    click.echo("\n=== Feature Views ===")
    for fv in feature_views:
        features_str = ", ".join([f"{f.name} ({f.dtype})" for f in fv.features])
        click.echo(f"  {fv.name}: features=[{features_str}], source={fv.source_path}")
        
    click.echo("\n=== On Demand Feature Views ===")
    for odfv in od_views:
        features_str = ", ".join([f"{f.name} ({f.dtype})" for f in odfv.features])
        click.echo(f"  {odfv.name}: features=[{features_str}], inputs={odfv.inputs}")

if __name__ == "__main__":
    main()
