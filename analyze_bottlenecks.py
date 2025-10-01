"""
Supply Chain Bottleneck Analysis - Simplified
==============================================
Purpose: Analyze % of late shipments affected by various bottlenecks
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def analyze_late_shipments(filepath):
    """Analyze percentage of late shipments by bottleneck factors."""
    
    # Load data
    print("\n" + "=" * 80)
    print("LATE SHIPMENT BOTTLENECK ANALYSIS")
    print("=" * 80 + "\n")
        
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset: {len(df):,} records")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}\n")
    
    # Define what "late" means (positive ETA variation)
    df['is_late'] = df['eta_variation_hours'] > 0
    overall_late_pct = df['is_late'].mean() * 100
    
    print(f"OVERALL: {overall_late_pct:.1f}% of shipments are late\n")
    print("=" * 80 + "\n")
        
    # Define bottleneck conditions (using 75th percentile for most - worst 25% of cases)
    bottlenecks = {
        'Long Lead Time (>75th percentile)': df['lead_time_days'] > df['lead_time_days'].quantile(0.75),
        'Slow Loading (>75th percentile)': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs (>75th percentile)': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'No Equipment Available': df['handling_equipment_availability'] == 0,
        'High Traffic Congestion (>75th pct)': df['traffic_congestion_level'] > df['traffic_congestion_level'].quantile(0.75),
        'High Port Congestion (>75th pct)': df['port_congestion_level'] > df['port_congestion_level'].quantile(0.75),
        'High Route Risk (>75th pct)': df['route_risk_level'] > df['route_risk_level'].quantile(0.75),
        'Severe Weather (>75th pct)': df['weather_condition_severity'] > df['weather_condition_severity'].quantile(0.75),
        'High Delay Probability (>75th pct)': df['delay_probability'] > df['delay_probability'].quantile(0.75),
        'Poor Cargo Condition': df['cargo_condition_status'] == 0,
        'Low Supplier Reliability (<25th pct)': df['supplier_reliability_score'] < df['supplier_reliability_score'].quantile(0.25),
        'High Fuel Consumption (>75th pct)': df['fuel_consumption_rate'] > df['fuel_consumption_rate'].quantile(0.75),
        'Extreme Temperature (>90th or <10th pct)': (df['iot_temperature'] < df['iot_temperature'].quantile(0.10)) | (df['iot_temperature'] > df['iot_temperature'].quantile(0.90)),
        'Order Not Fulfilled': df['order_fulfillment_status'] == 0,
    }
    
    # Calculate late % for each bottleneck
    results = []
    
    for bottleneck_name, condition in bottlenecks.items():
        affected_count = condition.sum()
        affected_pct = (affected_count / len(df)) * 100
        
        if affected_count > 0:
            late_when_affected = df[condition]['is_late'].sum()
            late_pct_when_affected = (late_when_affected / affected_count) * 100
            
            results.append({
                'Bottleneck': bottleneck_name,
                'Affected Shipments': affected_count,
                'Affected %': affected_pct,
                'Late When Affected': late_when_affected,
                'Late %': late_pct_when_affected
            })
    
    # Sort by affected percentage (descending)
    results.sort(key=lambda x: x['Affected %'], reverse=True)
    
    # Display results
    print("BOTTLENECK ANALYSIS - % OF LATE SHIPMENTS\n")
    print(f"{'Rank':<5} {'Bottleneck':<40} {'Affected':<12} {'Affected %':<12} {'Late Count':<12} {'Late %':<10}")
    print("-" * 100)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<5} {result['Bottleneck']:<40} {result['Affected Shipments']:<12,} {result['Affected %']:<11.1f}% {result['Late When Affected']:<12,} {result['Late %']:<9.1f}%")
    
    print("\nInterpretation:")
    print("- 'Affected %' = % of total shipments affected by this bottleneck")
    print("- 'Late %' = % of affected shipments that are late (ETA variation > 0)")
    print("- Compare 'Late %' to overall late rate of {:.1f}% to see impact".format(overall_late_pct))
    print("\nThreshold Methodology:")
    print("- Most bottlenecks use 75th percentile (worst 25% of cases)")
    print("- This ensures we capture meaningful outliers, not just average performance")
    print("- Binary variables (equipment, cargo) use actual status (0 = problem)\n")


def main():
    """Main execution function."""
    analyze_late_shipments('filtered_supply_chain_data.csv')


if __name__ == "__main__":
    main()
