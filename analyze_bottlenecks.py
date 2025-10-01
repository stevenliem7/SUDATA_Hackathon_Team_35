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
    
    # Define bottleneck conditions
    bottlenecks = {
        'Long Lead Time (>75th percentile)': df['lead_time_days'] > df['lead_time_days'].quantile(0.75),
        'Slow Loading (>75th percentile)': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs (>75th percentile)': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'No Equipment Available': df['handling_equipment_availability'] == 0,
        'High Traffic Congestion (>7)': df['traffic_congestion_level'] > 7,
        'High Port Congestion (>7)': df['port_congestion_level'] > 7,
        'High Route Risk (>7)': df['route_risk_level'] > 7,
        'Severe Weather (>0.7)': df['weather_condition_severity'] > 0.7,
        'High Delay Probability (>0.8)': df['delay_probability'] > 0.8,
        'Poor Cargo Condition': df['cargo_condition_status'] == 0,
        'Low Supplier Reliability (<0.3)': df['supplier_reliability_score'] < 0.3,
        'High Fuel Consumption (>15 L/h)': df['fuel_consumption_rate'] > 15,
        'Extreme Temperature': (df['iot_temperature'] < -10) | (df['iot_temperature'] > 40),
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
    
    print("\n" + "=" * 80)
    print("\nInterpretation:")
    print("- 'Affected %' = % of total shipments affected by this bottleneck")
    print("- 'Late %' = % of affected shipments that are late (ETA variation > 0)")
    print("- Compare 'Late %' to overall late rate of {:.1f}% to see impact\n".format(overall_late_pct))


def main():
    """Main execution function."""
    analyze_late_shipments('filtered_supply_chain_data.csv')


if __name__ == "__main__":
    main()
