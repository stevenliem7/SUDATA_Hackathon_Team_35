"""
Supply Chain Data Cleaning
Purpose: Filter data to Jan 2021 - Jan 2024 and aggregate to daily values
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_and_aggregate_data(input_file, output_file):
    """Clean data to the requested schema, then aggregate to daily values (Jan 2021 - Jan 2024)."""
    # Load raw dataset
    df = pd.read_csv(input_file)

    # 1) Timestamp — datetime (hourly)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # Standardize to hourly resolution (drop minutes/seconds if present)
    df['timestamp'] = df['timestamp'].dt.floor('H')

    # Original dataset summary
    print("\nOriginal Dataset Summary:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Column names: {', '.join(df.columns.tolist())}")

    # 2) Enforce types and ranges per requested schema
    # Floats
    float_columns_min_zero = [
        'fuel_consumption_rate',        # liters/hour
        'warehouse_inventory_level',    # units
        'loading_unloading_time',       # hours
        'shipping_costs',               # USD
        'lead_time_days',               # days
        'historical_demand',            # units
        'customs_clearance_time'        # hours/days (dataset-specific)
    ]

    float_columns_no_bounds = [
        'eta_variation_hours',          # hours (can be <0 or >0 depending on early/late definition)
        'iot_temperature',              # °C
        'delivery_time_deviation'       # generic float
    ]

    float_columns_0_to_1 = [
        'weather_condition_severity',
        'supplier_reliability_score',
        'driver_behavior_score',
        'fatigue_monitoring_score',
        'disruption_likelihood_score',
        'delay_probability'
    ]

    float_columns_0_to_10 = [
        'traffic_congestion_level',
        'port_congestion_level',
        'route_risk_level'
    ]

    # Convert to numeric and clamp ranges
    for col in float_columns_min_zero:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].clip(lower=0)

    for col in float_columns_no_bounds:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in float_columns_0_to_1:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, 1)

    for col in float_columns_0_to_10:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, 10)

    # 3) Binary columns — enforce int (0/1)
    binary_columns = [
        'handling_equipment_availability',
        'order_fulfillment_status',
        'cargo_condition_status'
    ]
    for col in binary_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Round decimals, then clip to [0,1] and cast to int
            df[col] = df[col].round().clip(0, 1).astype('Int64').astype(int)

    # 4) GPS coordinates
    if 'vehicle_gps_latitude' in df.columns:
        df['vehicle_gps_latitude'] = pd.to_numeric(df['vehicle_gps_latitude'], errors='coerce').clip(-90, 90)
    if 'vehicle_gps_longitude' in df.columns:
        df['vehicle_gps_longitude'] = pd.to_numeric(df['vehicle_gps_longitude'], errors='coerce').clip(-180, 180)

    # Filter to Jan 2021 - Jan 2024
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-01-31')
    df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    print(f"Filtered data: {len(df_filtered):,} records")
    print(f"Filtered range: {df_filtered['timestamp'].min().date()} to {df_filtered['timestamp'].max().date()}")

    # Create date column for daily aggregation
    df_filtered['date'] = df_filtered['timestamp'].dt.date

    # Columns to aggregate (numeric); keep the existing 22 core metrics if present
    numeric_columns = [
        'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
        'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
        'loading_unloading_time', 'weather_condition_severity', 'port_congestion_level',
        'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
        'historical_demand', 'iot_temperature', 'route_risk_level',
        'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
        'disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation'
    ]

    # Aggregate data by date
    daily_data = []
    for date, group in df_filtered.groupby('date'):
        daily_record = {'date': date}

        # For numeric columns, calculate mean (average daily value)
        for col in numeric_columns:
            if col in group.columns:
                daily_record[col] = group[col].mean()

        # For binary columns, calculate mean (percentage of 1s for the day)
        for col in binary_columns:
            if col in group.columns:
                daily_record[col] = group[col].mean()

        # Count of records for that day
        daily_record['daily_shipment_count'] = len(group)
        daily_data.append(daily_record)

    # Convert to DataFrame and finalize
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Basic data quality metrics (computed)
    print("Calculating data quality metrics...")
    total_cells = daily_df.shape[0] * daily_df.shape[1]
    null_cells = daily_df.isnull().sum().sum()
    completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells else 100.0

    # Simple validity snapshot
    validity_issues = 0
    total_values = 0
    numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'daily_shipment_count':
            total_values += len(daily_df[col])
            if col in ['fuel_consumption_rate', 'shipping_costs', 'warehouse_inventory_level',
                       'historical_demand', 'loading_unloading_time', 'customs_clearance_time']:
                validity_issues += (daily_df[col] < 0).sum()
            elif col in ['weather_condition_severity', 'supplier_reliability_score', 'driver_behavior_score',
                         'fatigue_monitoring_score', 'disruption_likelihood_score', 'delay_probability',
                         'handling_equipment_availability', 'order_fulfillment_status', 'cargo_condition_status']:
                validity_issues += ((daily_df[col] < 0) | (daily_df[col] > 1)).sum()
            elif col in ['traffic_congestion_level', 'port_congestion_level', 'route_risk_level']:
                validity_issues += ((daily_df[col] < 0) | (daily_df[col] > 10)).sum()
    validity = ((total_values - validity_issues) / total_values) * 100 if total_values else 100.0

    # Uniqueness (after aggregation, duplicates unlikely but we check)
    duplicates = daily_df.duplicated().sum()
    uniqueness = ((len(daily_df) - duplicates) / len(daily_df) * 100) if len(daily_df) else 100.0

    # Outlier detection (IQR-based snapshot)
    outlier_columns = ['fuel_consumption_rate', 'shipping_costs', 'warehouse_inventory_level',
                       'historical_demand', 'loading_unloading_time']
    total_outliers = 0
    total_checked = 0
    for col in outlier_columns:
        if col in daily_df.columns:
            Q1 = daily_df[col].quantile(0.25)
            Q3 = daily_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            total_outliers += ((daily_df[col] < lower) | (daily_df[col] > upper)).sum()
            total_checked += len(daily_df[col])
    outlier_pct = (total_outliers / total_checked * 100) if total_checked else 0.0

    # Save to CSV
    daily_df.to_csv(output_file, index=False)
    print(f"Saved daily data to: {output_file}")

    # Final dataset summary
    print("\nFinal Dataset Summary:")
    print(f"  Records: {len(daily_df):,}")
    print(f"  Columns: {len(daily_df.columns)}")
    print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")

    # Data quality metrics
    print("\nData Quality Metrics:")
    print(f"  Completeness: {completeness:.2f}%  (non-null coverage)")
    print(f"  Validity: {validity:.2f}%  (in-range values)")
    print(f"  Uniqueness: {uniqueness:.2f}%  (deduplication check)")
    print(f"  Outlier: {outlier_pct:.2f}%  (IQR-based snapshot)")

    return daily_df


def main():
    """Main function to clean and aggregate the data."""
    input_file = 'dynamic_supply_chain_logistics_dataset.csv'
    output_file = 'cleaned_supply_chain_logistics_dataset.csv'

    _ = clean_and_aggregate_data(input_file, output_file)


if __name__ == "__main__":
    main()