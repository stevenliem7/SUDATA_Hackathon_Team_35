"""
Supply Chain Bottleneck Analysis
Purpose: Analyze % of late shipments affected by various bottlenecks
"""
import pandas as pd

def analyze_late_shipments(filepath):
    """Analyze percentage of late shipments by bottleneck factors."""
    
    # Load data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Dataset: {len(df):,} records")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}\n")
    
    # Late shipment means positive ETA variation 
    df['is_late'] = df['eta_variation_hours'] > 0
    overall_late_pct = df['is_late'].mean() * 100
    print(f"OVERALL: {overall_late_pct:.1f}% of shipments are late\n")
    
    # Define bottleneck conditions (using 75th percentile for most - worst 25% of cases)
    bottlenecks = {
        'Long Lead Time': df['lead_time_days'] > df['lead_time_days'].quantile(0.75),
        'Slow Loading': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'No Equipment': df['handling_equipment_availability'] == 0,
        'High Traffic': df['traffic_congestion_level'] > df['traffic_congestion_level'].quantile(0.75),
        'High Port Congestion': df['port_congestion_level'] > df['port_congestion_level'].quantile(0.75),
        'High Route Risk': df['route_risk_level'] > df['route_risk_level'].quantile(0.75),
        'Severe Weather': df['weather_condition_severity'] > df['weather_condition_severity'].quantile(0.75),
        'High Delay Probability': df['delay_probability'] > df['delay_probability'].quantile(0.75),
        'Poor Cargo': df['cargo_condition_status'] == 0,
        'Low Supplier Reliability': df['supplier_reliability_score'] < df['supplier_reliability_score'].quantile(0.25),
        'High Fuel Consumption': df['fuel_consumption_rate'] > df['fuel_consumption_rate'].quantile(0.75),
        'Extreme Temperature': (df['iot_temperature'] < df['iot_temperature'].quantile(0.10)) | (df['iot_temperature'] > df['iot_temperature'].quantile(0.90)),
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
                'Affected %': affected_pct,
                'Late %': late_pct_when_affected
            })
    
    # Sort by affected percentage (descending)
    results.sort(key=lambda x: x['Affected %'], reverse=True)
    
    # Display compact results table
    print("BOTTLENECK ANALYSIS - % OF LATE SHIPMENTS\n")
    print(f"{'Rank':<4} {'Bottleneck':<20} {'Affected %':<10} {'Late %':<8}")
    print("-" * 45)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result['Bottleneck']:<20} {result['Affected %']:<9.1f}% {result['Late %']:<7.1f}%")

def main():
    analyze_late_shipments('cleaned_supply_chain_logistics_dataset.csv')
if __name__ == "__main__":
    main()