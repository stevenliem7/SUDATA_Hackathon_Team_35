"""
External Network Stress Index Analysis
=====================================
Purpose: Analyze how the number of external bottlenecks affects ETA variation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def analyze_operational_stress(filepath):
    """Analyze Operational Stress Index vs late deliveries."""
    
    print("\n" + "=" * 80)
    print("OPERATIONAL STRESS INDEX ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Dataset: {len(df):,} records")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}\n")
    
    # Define the 6 operational stress factors
    stress_factors = {
        'Slow Loading': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'High Traffic': df['traffic_congestion_level'] > df['traffic_congestion_level'].quantile(0.75),
        'High Port Congestion': df['port_congestion_level'] > df['port_congestion_level'].quantile(0.75),
        'High Route Risk': df['route_risk_level'] > df['route_risk_level'].quantile(0.75),
        'Severe Weather': df['weather_condition_severity'] > df['weather_condition_severity'].quantile(0.75)
    }
    
    # Calculate stress index (number of bottlenecks per shipment)
    df['stress_index'] = 0
    for factor_name, condition in stress_factors.items():
        df['stress_index'] += condition.astype(int)
    
    # Analyze relationship between stress index and late deliveries
    print("OPERATIONAL STRESS INDEX vs LATE DELIVERIES\n")
    print("=" * 80)
    
    # Create late delivery flag (ETA variation above median for daily aggregated data)
    median_eta = df['eta_variation_hours'].median()
    df['is_late'] = df['eta_variation_hours'] > median_eta
    
    # Group by stress index and calculate statistics
    stress_analysis = df.groupby('stress_index').agg({
        'is_late': ['count', 'sum', 'mean'],
        'order_fulfillment_status': 'mean'
    }).round(2)
    
    # Flatten column names
    stress_analysis.columns = ['Shipments', 'Late_Deliveries', 'Late_Percentage', 'Fulfillment_Rate']
    stress_analysis['Late_Percentage'] = stress_analysis['Late_Percentage'] * 100
    stress_analysis['Fulfillment_Rate'] = stress_analysis['Fulfillment_Rate'] * 100
    
    # Add percentage of total shipments
    stress_analysis['Pct_of_Total'] = (stress_analysis['Shipments'] / len(df) * 100).round(1)
    
    print(f"{'Stress':<8} {'Shipments':<12} {'% Total':<8} {'Late Count':<12} {'Late %':<8} {'Fulfill%':<10}")
    print("-" * 70)
    
    for stress_level in range(7):  # 0 to 6 bottlenecks
        if stress_level in stress_analysis.index:
            row = stress_analysis.loc[stress_level]
            print(f"{stress_level:<8} {row['Shipments']:<12,} {row['Pct_of_Total']:<7.1f}% {row['Late_Deliveries']:<12,} {row['Late_Percentage']:<7.1f}% {row['Fulfillment_Rate']:<9.1f}%")
        else:
            print(f"{stress_level:<8} {'0':<12} {'0.0':<7}% {'0':<12} {'0.0':<7}% {'N/A':<9}")
    
    print("\n" + "=" * 80)
    print("\nKEY INSIGHTS:\n")
    
    # Calculate correlation
    correlation = df['stress_index'].corr(df['is_late'])
    print(f"• Correlation between Stress Index and Late Deliveries: {correlation:.3f}")

    # Weighted correlation by shipments across stress_index levels
    try:
        grouped = stress_analysis.reset_index()[['stress_index', 'Shipments', 'Late_Percentage']].copy()
        grouped['Late_Rate'] = grouped['Late_Percentage'] / 100.0
        w = grouped['Shipments'].astype(float).values
        x = grouped['stress_index'].astype(float).values
        y = grouped['Late_Rate'].astype(float).values
        if w.sum() > 0 and len(grouped) > 1:
            wx = (w * x).sum() / w.sum()
            wy = (w * y).sum() / w.sum()
            cov_w = (w * (x - wx) * (y - wy)).sum() / w.sum()
            varx_w = (w * (x - wx) ** 2).sum() / w.sum()
            vary_w = (w * (y - wy) ** 2).sum() / w.sum()
            weighted_corr = cov_w / np.sqrt(varx_w * vary_w) if varx_w > 0 and vary_w > 0 else np.nan
        else:
            weighted_corr = np.nan
        print(f"• Weighted correlation (by shipments): {weighted_corr:.3f}")
    except Exception:
        pass
    
    # Compare 0 vs 3+ stress factors
    zero_stress_late_pct = df[df['stress_index'] == 0]['is_late'].mean() * 100
    high_stress_late_pct = df[df['stress_index'] >= 3]['is_late'].mean() * 100
    
    if not pd.isna(zero_stress_late_pct) and not pd.isna(high_stress_late_pct):
        print(f"• Late delivery rate with 0 stress factors: {zero_stress_late_pct:.1f}%")
        print(f"• Late delivery rate with 3+ stress factors: {high_stress_late_pct:.1f}%")
        print(f"• Impact of high stress: +{high_stress_late_pct - zero_stress_late_pct:.1f} percentage points")
    
    # Fulfillment rate impact
    zero_fulfill = df[df['stress_index'] == 0]['order_fulfillment_status'].mean() * 100
    high_fulfill = df[df['stress_index'] >= 3]['order_fulfillment_status'].mean() * 100
    
    if not pd.isna(zero_fulfill) and not pd.isna(high_fulfill):
        print(f"• Fulfillment rate with 0 stress factors: {zero_fulfill:.1f}%")
        print(f"• Fulfillment rate with 3+ stress factors: {high_fulfill:.1f}%")
        print(f"• Fulfillment impact: {zero_fulfill - high_fulfill:.1f} percentage points")
    
    print("\n" + "=" * 80)
    print("\nINDIVIDUAL STRESS FACTOR CONTRIBUTION:\n")
    
    # Analyze each factor's impact on late deliveries
    factor_impacts = []
    for factor_name, condition in stress_factors.items():
        with_factor_late_pct = df[condition]['is_late'].mean() * 100
        without_factor_late_pct = df[~condition]['is_late'].mean() * 100
        impact = with_factor_late_pct - without_factor_late_pct
        frequency = condition.sum() / len(df) * 100
        
        factor_impacts.append({
            'Factor': factor_name,
            'Impact': impact,
            'Frequency': frequency,
            'Late_With': with_factor_late_pct,
            'Late_Without': without_factor_late_pct
        })
    
    # Sort by impact
    factor_impacts.sort(key=lambda x: x['Impact'], reverse=True)
    
    print(f"{'Factor':<25} {'Impact':<10} {'Frequency':<12} {'Late With':<10} {'Late Without':<12}")
    print("-" * 75)
    
    for factor in factor_impacts:
        print(f"{factor['Factor']:<25} {factor['Impact']:>+8.1f}% {factor['Frequency']:>10.1f}% {factor['Late_With']:>8.1f}% {factor['Late_Without']:>10.1f}%")
    
    print("\n" + "=" * 80)
    print("\nRECOMMENDATIONS:\n")
    
    # Find the most impactful factors
    top_impact = factor_impacts[0]
    print(f"• Primary stress factor: {top_impact['Factor']} (+{top_impact['Impact']:.1f}% late delivery impact)")
    print(f"• Secondary stress factor: {factor_impacts[1]['Factor']} (+{factor_impacts[1]['Impact']:.1f}% late delivery impact)")
    
    # Stress level recommendations
    high_stress_count = (df['stress_index'] >= 3).sum()
    high_stress_pct = high_stress_count / len(df) * 100
    
    print(f"• {high_stress_pct:.1f}% of shipments face 3+ stress factors")
    print(f"• Focus on reducing the most frequent high-impact factors")
    print(f"• Consider stress factor combinations in route planning")
    
    return df


def main():
    """Main execution function."""
    df = analyze_operational_stress('cleaned_supply_chain_logistics_dataset.csv')
    
    # Save results summary
    print("\n" + "=" * 80)
    print("SAVING ANALYSIS SUMMARY")
    print("=" * 80 + "\n")
    
    with open('operational_stress_summary.txt', 'w') as f:
        f.write("OPERATIONAL STRESS INDEX SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall stats
        f.write(f"Total Shipments: {len(df):,}\n")
        f.write(f"Average Stress Index: {df['stress_index'].mean():.2f}\n")
        f.write(f"Overall Late Delivery Rate: {df['is_late'].mean()*100:.1f}%\n")
        f.write(f"Correlation (Stress vs Late Deliveries): {df['stress_index'].corr(df['is_late']):.3f}\n\n")
        
        # Stress distribution
        f.write("STRESS INDEX DISTRIBUTION:\n")
        stress_dist = df['stress_index'].value_counts().sort_index()
        for stress_level, count in stress_dist.items():
            pct = count / len(df) * 100
            f.write(f"  {stress_level} factors: {count:,} shipments ({pct:.1f}%)\n")
    
    print("Analysis summary saved: operational_stress_summary.txt")


if __name__ == "__main__":
    main()