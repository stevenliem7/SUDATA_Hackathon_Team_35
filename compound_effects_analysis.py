import pandas as pd
import numpy as np

print('\nCOMPOUND EFFECTS ANALYSIS - DELIVERY COST & TIME')
print('Proving weak correlations hide strong compound impacts\n')

df = pd.read_csv('filtered_supply_chain_data.csv')
print(f'Loaded {len(df):,} records\n')

# Create 8 bottleneck indicators
print('Creating bottleneck indicators...')
df['bn_late'] = (df['eta_variation_hours'] > 3).astype(int)
df['bn_no_equip'] = (df['handling_equipment_availability'] == 0).astype(int)
df['bn_traffic'] = (df['traffic_congestion_level'] > 7).astype(int)
df['bn_slow_load'] = (df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75)).astype(int)
df['bn_slow_customs'] = (df['customs_clearance_time'] > df['customs_clearance_time'].quantile(0.75)).astype(int)
df['bn_long_lead'] = (df['lead_time_days'] > df['lead_time_days'].quantile(0.75)).astype(int)
df['bn_port_congestion'] = (df['port_congestion_level'] > 7).astype(int)
df['bn_high_risk'] = (df['route_risk_level'] > 7).astype(int)

df['total_bn'] = df[['bn_late', 'bn_no_equip', 'bn_traffic', 'bn_slow_load', 
                      'bn_slow_customs', 'bn_long_lead', 'bn_port_congestion', 'bn_high_risk']].sum(axis=1)

print(f'Created 8 bottleneck indicators')
print(f'Average: {df["total_bn"].mean():.2f}, Max: {df["total_bn"].max()}\n')

print('DISTRIBUTION:')
for i in range(int(df['total_bn'].max()) + 1):
    count = (df['total_bn'] == i).sum()
    pct = count / len(df) * 100
    print(f'  {i} bottlenecks: {count:>6,} ({pct:>5.1f}%)')

print('\nCOMPOUND EFFECT ON DELIVERY TIME\n')
time_stats = df.groupby('total_bn').agg({
    'eta_variation_hours': ['count', 'mean'],
    'order_fulfillment_status': 'mean',
    'shipping_costs': 'mean'
}).round(2)

print('Bottlenecks  Count      %     AvgDelay  Fulfill%   AvgCost')
print('-' * 70)
for idx in sorted(df['total_bn'].unique()):
    count = int(time_stats.loc[idx, ('eta_variation_hours', 'count')])
    pct = count / len(df) * 100
    delay = time_stats.loc[idx, ('eta_variation_hours', 'mean')]
    fulfill = time_stats.loc[idx, ('order_fulfillment_status', 'mean')] * 100
    cost = time_stats.loc[idx, ('shipping_costs', 'mean')]
    print(f'{idx:9} {count:>7,} {pct:>5.1f}% {delay:>7.2f}h {fulfill:>7.1f}% ${cost:>7.2f}')

zero_delay = df[df['total_bn'] == 0]['eta_variation_hours'].mean()
max_bn = int(df['total_bn'].max())
max_delay = df[df['total_bn'] == max_bn]['eta_variation_hours'].mean()
print(f'\nImpact: {zero_delay:.2f}h â†’ {max_delay:.2f}h ({((max_delay/zero_delay)-1)*100:.0f}% increase)\n')

# NEW SECTION: Individual Bottleneck Impact Analysis
print('=' * 80)
print('INDIVIDUAL BOTTLENECK IMPACT ON AVERAGE DELAY')
print('=' * 80 + '\n')

bottleneck_impact = []
bn_cols = ['bn_late', 'bn_no_equip', 'bn_traffic', 'bn_slow_load', 
           'bn_slow_customs', 'bn_long_lead', 'bn_port_congestion', 'bn_high_risk']

bn_names = {
    'bn_late': 'Late Delivery (>3h)',
    'bn_no_equip': 'No Equipment Available',
    'bn_traffic': 'High Traffic (>7/10)',
    'bn_slow_load': 'Slow Loading (Top 25%)',
    'bn_slow_customs': 'Slow Customs (Top 25%)',
    'bn_long_lead': 'Long Lead Time (Top 25%)',
    'bn_port_congestion': 'Port Congestion (>7/10)',
    'bn_high_risk': 'High Route Risk (>7/10)'
}

print('When Each Bottleneck is Present vs Absent:\n')
print(f'{'Bottleneck':35} {'Present':>10} {'Absent':>10} {'Impact':>10} {'Corr':>8} {'Freq':>8}')
print('-' * 85)

for bn in bn_cols:
    with_bn = df[df[bn] == 1]['eta_variation_hours'].mean()
    without_bn = df[df[bn] == 0]['eta_variation_hours'].mean()
    impact = with_bn - without_bn
    corr = df[bn].corr(df['eta_variation_hours'])
    freq = (df[bn] == 1).sum() / len(df) * 100
    
    bottleneck_impact.append({
        'name': bn_names[bn],
        'with': with_bn,
        'without': without_bn,
        'impact': impact,
        'corr': corr,
        'freq': freq
    })
    
    print(f'{bn_names[bn]:35} {with_bn:>9.2f}h {without_bn:>9.2f}h {impact:>9.2f}h {corr:>7.3f} {freq:>7.1f}%')

# Sort by impact
impact_df = pd.DataFrame(bottleneck_impact).sort_values('impact', ascending=False)

print(f'\nRANKED BY DELAY IMPACT (Highest to Lowest):\n')
for i, row in impact_df.iterrows():
    print(f'{i+1}. {row["name"]:35} +{row["impact"]:.2f}h delay when present')

print('\nSTATISTICAL PROOF\n')
single_corrs = []
for var in ['loading_unloading_time', 'traffic_congestion_level', 'customs_clearance_time', 'lead_time_days']:
    corr = df[var].corr(df['eta_variation_hours'])
    single_corrs.append(corr)
    print(f'{var:30s} -> Delay: r={corr:+.4f}')

compound_corr = df['total_bn'].corr(df['eta_variation_hours'])
avg_single = np.mean([abs(c) for c in single_corrs])
print(f'\ntotal_bottlenecks -> Delay: r={compound_corr:+.4f}')
print(f'Compound is {abs(compound_corr)/avg_single:.1f}x STRONGER!\n')

clean_fulfill = df[df['total_bn'] == 0]['order_fulfillment_status'].mean() * 100
severe_fulfill = df[df['total_bn'] >= 5]['order_fulfillment_status'].mean() * 100
clean_cost = df[df['total_bn'] == 0]['shipping_costs'].mean()
severe_cost = df[df['total_bn'] >= 5]['shipping_costs'].mean()
clean_delay = df[df['total_bn'] == 0]['eta_variation_hours'].mean()
severe_delay = df[df['total_bn'] >= 5]['eta_variation_hours'].mean()

print('CONCLUSION:')
print(f'Clean (0 bn): {clean_fulfill:.1f}% fulfill, ${clean_cost:.2f} cost, {clean_delay:.2f}h delay')
print(f'Severe (5+ bn): {severe_fulfill:.1f}% fulfill, ${severe_cost:.2f} cost, {severe_delay:.2f}h delay')
print(f'\nImpact: {abs(clean_fulfill-severe_fulfill):.1f}pp fulfillment, ${abs(severe_cost-clean_cost):.2f} cost, +{severe_delay-clean_delay:.2f}h delay')
print(f'\n{(df["total_bn"] >= 3).sum():,} deliveries ({(df["total_bn"] >= 3).sum()/len(df)*100:.1f}%) have 3+ bottlenecks')
print('Weak correlations mask SYSTEMIC compound inefficiency!\n')
