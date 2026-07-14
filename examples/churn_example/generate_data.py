import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_churn_dataset(dest_dir: str):
    """
    Generates a realistic customer churn dataset.
    Creates profile features and time-series activity features, saved as Parquet files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Generate Customers
    num_customers = 500
    customer_ids = np.arange(10001, 10001 + num_customers)
    
    # 2. Customer Profile Data (static features or features changing very slowly)
    signup_dates = [datetime(2026, 1, 1) + timedelta(days=int(np.random.randint(0, 100))) for _ in range(num_customers)]
    plan_types = np.random.choice(["Basic", "Premium", "Enterprise"], size=num_customers, p=[0.5, 0.3, 0.2])
    regions = np.random.choice(["US", "EU", "APAC"], size=num_customers)
    
    profile_df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "plan_type": plan_types,
        "region": regions
    })
    profile_df.to_parquet(os.path.join(dest_dir, "customer_profiles.parquet"))
    print(f"Generated {num_customers} customer profiles.")

    # 3. Customer Activity Data (dynamic features updated daily/weekly)
    # Generate activity updates over a 30-day period (e.g. June 1 to June 30, 2026)
    start_date = datetime(2026, 6, 1)
    activity_records = []
    
    for customer_id in customer_ids:
        # Every customer has weekly activity updates
        current_date = start_date
        for week in range(4):
            timestamp = current_date + timedelta(days=int(np.random.randint(0, 5)), hours=int(np.random.randint(0, 24)))
            
            # Simulate features
            usage_minutes = float(np.random.normal(loc=120.0, scale=30.0) + (100.0 if plan_types[customer_id - 10001] == "Premium" else 0))
            support_calls = int(np.random.poisson(lam=0.5 if plan_types[customer_id - 10001] == "Basic" else 0.1))
            active_days_in_week = int(np.random.randint(1, 8))
            
            activity_records.append({
                "customer_id": customer_id,
                "timestamp": timestamp.isoformat(),
                "usage_minutes": max(0.0, usage_minutes),
                "support_calls": support_calls,
                "active_days_in_week": active_days_in_week
            })
            current_date += timedelta(days=7)

    activity_df = pd.DataFrame(activity_records)
    activity_df.to_parquet(os.path.join(dest_dir, "customer_activities.parquet"))
    print(f"Generated {len(activity_df)} weekly activity records.")

    # 4. Generate Churn Event Labels (label times for training predictions)
    # For point-in-time correct join testing: we want label time to be on June 28, 2026
    label_records = []
    for customer_id in customer_ids:
        label_time = datetime(2026, 6, 28, 12, 0, 0)
        # Churn probability higher if support calls are high or usage minutes are low
        idx = customer_id - 10001
        is_churn = 0
        if plan_types[idx] == "Basic" and np.random.rand() > 0.6:
            is_churn = 1
        elif np.random.rand() > 0.95:
            is_churn = 1

        label_records.append({
            "customer_id": customer_id,
            "timestamp": label_time.isoformat(),
            "churned": is_churn
        })

    labels_df = pd.DataFrame(label_records)
    labels_df.to_parquet(os.path.join(dest_dir, "churn_labels.parquet"))
    print(f"Generated {len(labels_df)} churn labels for point-in-time training.")

if __name__ == "__main__":
    generate_churn_dataset(os.path.dirname(__file__))
