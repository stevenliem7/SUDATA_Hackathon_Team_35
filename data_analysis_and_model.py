"""
External Network Stress Index Analysis
=====================================
Purpose: Analyze how the number of external bottlenecks affects ETA variation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def analyze_operational_stress(filepath):
    """Analyze Operational Stress Index vs late deliveries."""
    
    # Load data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define the 6 operational stress factors as continuous severity scores (0-1)
    def normalize_score(series, higher_worse=True):
        """Normalize series to 0-1 scale where higher values indicate more stress."""
        s = pd.to_numeric(series, errors='coerce')
        min_val, max_val = s.min(), s.max()
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            return pd.Series(0.0, index=s.index)
        normalized = (s - min_val) / (max_val - min_val)
        if not higher_worse:  # For factors where lower values are worse (e.g., cargo condition)
            normalized = 1 - normalized
        return normalized.fillna(0.0)
    
    # Create continuous severity scores (0 = no stress, 1 = maximum stress)
    df['slow_loading_score'] = normalize_score(df['loading_unloading_time'], higher_worse=True)
    df['slow_customs_score'] = normalize_score(df['customs_clearance_time'], higher_worse=True)
    df['high_traffic_score'] = normalize_score(df['traffic_congestion_level'], higher_worse=True)
    df['high_port_congestion_score'] = normalize_score(df['port_congestion_level'], higher_worse=True)
    df['high_route_risk_score'] = normalize_score(df['route_risk_level'], higher_worse=True)
    df['severe_weather_score'] = normalize_score(df['weather_condition_severity'], higher_worse=True)
    
    # Calculate continuous stress index (sum of normalized scores)
    df['stress_index'] = (df['slow_loading_score'] + 
                         df['slow_customs_score'] + 
                         df['high_traffic_score'] + 
                         df['high_port_congestion_score'] + 
                         df['high_route_risk_score'] + 
                         df['severe_weather_score'])
    
    # Also create discrete stress levels for comparison
    stress_factors = {
        'Slow Loading': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'High Traffic': df['traffic_congestion_level'] > df['traffic_congestion_level'].quantile(0.75),
        'High Port Congestion': df['port_congestion_level'] > df['port_congestion_level'].quantile(0.75),
        'High Route Risk': df['route_risk_level'] > df['route_risk_level'].quantile(0.75),
        'Severe Weather': df['weather_condition_severity'] > df['weather_condition_severity'].quantile(0.75)
    }
    
    # Calculate discrete stress index (count of binary flags)
    df['discrete_stress_index'] = 0
    for factor_name, condition in stress_factors.items():
        df['discrete_stress_index'] += condition.astype(int)
    
    # Analyze relationship between stress index and late deliveries
    
    # Create late delivery flag (ETA variation above median for daily aggregated data)
    median_eta = df['eta_variation_hours'].median()
    df['is_late'] = df['eta_variation_hours'] > median_eta
    
    # Group by discrete stress index and calculate statistics
    stress_analysis = df.groupby('discrete_stress_index').agg({
        'is_late': ['count', 'sum', 'mean'],
        'order_fulfillment_status': 'mean'
    }).round(2)
    
    # Flatten column names
    stress_analysis.columns = ['Shipments', 'Late_Deliveries', 'Late_Percentage', 'Fulfillment_Rate']
    stress_analysis['Late_Percentage'] = stress_analysis['Late_Percentage'] * 100
    stress_analysis['Fulfillment_Rate'] = stress_analysis['Fulfillment_Rate'] * 100
    
    # Add percentage of total shipments
    stress_analysis['Pct_of_Total'] = (stress_analysis['Shipments'] / len(df) * 100).round(1)
    
    # Calculate correlation
    correlation = df['discrete_stress_index'].corr(df['is_late'])
    
    # Weighted correlation by shipments across discrete_stress_index levels
    try:
        grouped = stress_analysis.reset_index()[['discrete_stress_index', 'Shipments', 'Late_Percentage']].copy()
        grouped['Late_Rate'] = grouped['Late_Percentage'] / 100.0
        w = grouped['Shipments'].astype(float).values
        x = grouped['discrete_stress_index'].astype(float).values
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
    except Exception:
        weighted_corr = np.nan
    
    # Compare 0 vs 3+ stress factors
    zero_stress_late_pct = df[df['discrete_stress_index'] == 0]['is_late'].mean() * 100
    high_stress_late_pct = df[df['discrete_stress_index'] >= 3]['is_late'].mean() * 100
    
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
    
    # Find the most impactful factors
    top_impact = factor_impacts[0]
    high_stress_count = (df['discrete_stress_index'] >= 3).sum()
    high_stress_pct = high_stress_count / len(df) * 100
    
    # Console output for stress factor quantities vs late delivery rate
    print("STRESS FACTOR QUANTITIES vs LATE DELIVERY RATE")
    print("=" * 50)
    print(f"{'Stress':<8} {'Shipments':<12} {'% Total':<8} {'Late Count':<12} {'Late %':<8}")
    print("-" * 50)
    
    for stress_level in range(7):  # 0 to 6 bottlenecks
        if stress_level in stress_analysis.index:
            row = stress_analysis.loc[stress_level]
            print(f"{stress_level:<8} {row['Shipments']:<12,} {row['Pct_of_Total']:<7.1f}% {row['Late_Deliveries']:<12,} {row['Late_Percentage']:<7.1f}%")
        else:
            print(f"{stress_level:<8} {'0':<12} {'0.0':<7}% {'0':<12} {'0.0':<7}%")
    
    print("\nINDIVIDUAL STRESS FACTOR IMPACT")
    print("=" * 50)
    print(f"{'Factor':<25} {'Impact':<10} {'Frequency':<12} {'Late With':<10} {'Late Without':<12}")
    print("-" * 70)
    
    for factor in factor_impacts:
        print(f"{factor['Factor']:<25} {factor['Impact']:>+8.1f}% {factor['Frequency']:>10.1f}% {factor['Late_With']:>8.1f}% {factor['Late_Without']:>10.1f}%")
    
    return df


def create_stress_analysis_graphs(df):
    """Create comprehensive graphs analyzing stress factors."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set smaller font sizes (40% reduction)
    plt.rcParams.update({
        'font.size': 8,           # Base font size (was ~13)
        'axes.titlesize': 9,      # Title font size (was ~15)
        'axes.labelsize': 8,      # Axis label font size (was ~13)
        'xtick.labelsize': 7,     # X-axis tick font size (was ~12)
        'ytick.labelsize': 7,     # Y-axis tick font size (was ~12)
        'legend.fontsize': 7,     # Legend font size (was ~12)
        'figure.titlesize': 10    # Figure title font size (was ~16)
    })
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Stress Index Distribution
    plt.subplot(3, 3, 1)
    stress_counts = df['discrete_stress_index'].value_counts().sort_index()
    bars = plt.bar(stress_counts.index, stress_counts.values, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Number of Stress Factors')
    plt.ylabel('Number of Days')
    plt.title('Distribution of Stress Factors')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Late Delivery Rate by Stress Level
    plt.subplot(3, 3, 2)
    stress_late = df.groupby('discrete_stress_index')['is_late'].mean() * 100
    plt.plot(stress_late.index, stress_late.values, marker='o', linewidth=2, markersize=8, color='red')
    plt.xlabel('Number of Stress Factors')
    plt.ylabel('Late Delivery Rate (%)')
    plt.title('Late Delivery Rate vs Stress Factors')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(stress_late.index, stress_late.values)):
        plt.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom')
    
    # 3. Fulfillment Rate by Stress Level
    plt.subplot(3, 3, 3)
    stress_fulfill = df.groupby('discrete_stress_index')['order_fulfillment_status'].mean() * 100
    plt.plot(stress_fulfill.index, stress_fulfill.values, marker='s', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Stress Factors')
    plt.ylabel('Fulfillment Rate (%)')
    plt.title('Fulfillment Rate vs Stress Factors')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(stress_fulfill.index, stress_fulfill.values)):
        plt.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom')
    
    # 4. Individual Stress Factor Impact
    plt.subplot(3, 3, 4)
    factor_impacts = []
    stress_factors = {
        'Slow Loading': df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75),
        'Slow Customs': df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75),
        'High Traffic': df['traffic_congestion_level'] > df['traffic_congestion_level'].quantile(0.75),
        'High Port Congestion': df['port_congestion_level'] > df['port_congestion_level'].quantile(0.75),
        'High Route Risk': df['route_risk_level'] > df['route_risk_level'].quantile(0.75),
        'Severe Weather': df['weather_condition_severity'] > df['weather_condition_severity'].quantile(0.75)
    }
    
    for factor_name, condition in stress_factors.items():
        with_factor_late_pct = df[condition]['is_late'].mean() * 100
        without_factor_late_pct = df[~condition]['is_late'].mean() * 100
        impact = with_factor_late_pct - without_factor_late_pct
        factor_impacts.append((factor_name, impact))
    
    factor_impacts.sort(key=lambda x: x[1], reverse=True)
    factors, impacts = zip(*factor_impacts)
    
    bars = plt.barh(range(len(factors)), impacts, alpha=0.7, color='orange', edgecolor='darkorange')
    plt.yticks(range(len(factors)), [f.replace(' ', '\n') for f in factors])
    plt.xlabel('Impact on Late Delivery Rate (%)')
    plt.title('Individual Stress Factor Impact')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, impact) in enumerate(zip(bars, impacts)):
        plt.text(impact + (0.1 if impact >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                f'{impact:+.1f}%', ha='left' if impact >= 0 else 'right', va='center')
    
    # 5. Stress Factor Frequency
    plt.subplot(3, 3, 5)
    factor_frequencies = []
    for factor_name, condition in stress_factors.items():
        frequency = condition.sum() / len(df) * 100
        factor_frequencies.append((factor_name, frequency))
    
    factor_frequencies.sort(key=lambda x: x[1], reverse=True)
    factors, frequencies = zip(*factor_frequencies)
    
    bars = plt.barh(range(len(factors)), frequencies, alpha=0.7, color='purple', edgecolor='indigo')
    plt.yticks(range(len(factors)), [f.replace(' ', '\n') for f in factors])
    plt.xlabel('Frequency (%)')
    plt.title('Stress Factor Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        plt.text(freq + 0.5, bar.get_y() + bar.get_height()/2,
                f'{freq:.1f}%', ha='left', va='center')
    
    # 6. ETA Variation Distribution
    plt.subplot(3, 3, 6)
    plt.hist(df['eta_variation_hours'], bins=30, alpha=0.7, color='teal', edgecolor='darkcyan')
    plt.axvline(df['eta_variation_hours'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    plt.xlabel('ETA Variation (hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ETA Variation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Correlation Heatmap
    plt.subplot(3, 3, 7)
    # Select key columns for correlation
    corr_cols = ['discrete_stress_index', 'is_late', 'order_fulfillment_status', 
                 'loading_unloading_time', 'customs_clearance_time', 'traffic_congestion_level',
                 'port_congestion_level', 'route_risk_level', 'weather_condition_severity']
    corr_data = df[corr_cols].corr()
    
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix')
    
    # 8. Stress Index vs ETA Variation Scatter
    plt.subplot(3, 3, 8)
    plt.scatter(df['discrete_stress_index'], df['eta_variation_hours'], alpha=0.6, color='coral', s=20)
    plt.xlabel('Stress Index')
    plt.ylabel('ETA Variation (hours)')
    plt.title('Stress Index vs ETA Variation')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['discrete_stress_index'], df['eta_variation_hours'], 1)
    p = np.poly1d(z)
    plt.plot(df['discrete_stress_index'], p(df['discrete_stress_index']), "r--", alpha=0.8, linewidth=2)
    
    # 9. Monthly Trend Analysis
    plt.subplot(3, 3, 9)
    df['month'] = df['date'].dt.to_period('M')
    monthly_stress = df.groupby('month')['discrete_stress_index'].mean()
    monthly_late = df.groupby('month')['is_late'].mean() * 100
    
    ax1 = plt.gca()
    color = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Stress Index', color=color)
    line1 = ax1.plot(range(len(monthly_stress)), monthly_stress.values, color=color, marker='o', label='Stress Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Late Delivery Rate (%)', color=color)
    line2 = ax2.plot(range(len(monthly_late)), monthly_late.values, color=color, marker='s', label='Late Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set x-axis labels
    plt.xticks(range(0, len(monthly_stress), 6), 
               [str(monthly_stress.index[i]) for i in range(0, len(monthly_stress), 6)], 
               rotation=45)
    
    plt.title('Monthly Trends: Stress vs Late Deliveries')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('stress_analysis_graphs.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    df = analyze_operational_stress('cleaned_supply_chain_logistics_dataset.csv')
    
    # Create comprehensive graphs
    create_stress_analysis_graphs(df)
    


if __name__ == "__main__":
    main()