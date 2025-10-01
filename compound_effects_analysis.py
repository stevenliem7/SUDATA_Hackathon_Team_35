import pandas as pd
import numpy as np

print('\nCOMPOUND EFFECTS ANALYSIS - DELIVERY COST & TIME')
print('Proving weak correlations hide strong compound impacts\n')

df = pd.read_csv('filtered_supply_chain_data.csv')
print(f'Loaded {len(df):,} records\n')

df['bn_late'] = (df['eta_variation_hours'] > 3).astype(int)
df['bn_no_equip'] = (df['handling_equipment_availability'] == 0).astype(int)
df['bn_traffic'] = (df['traffic_congestion_level'] > 7).astype(int)
df['bn_slow_load'] = (df['loading_unloading_time'] > df['loading_unloading_time'].quantile(0.75)).astype(int)
df['total_bn'] = df[['bn_late', 'bn_no_equip', 'bn_traffic', 'bn_slow_load']].sum(axis=1)

print(f'Created bottleneck indicators\n')

print('COMPOUND EFFECT ON DELIVERY TIME\n')
time_stats = df.groupby('total_bn').agg({'eta_variation_hours': ['count', 'mean'], 'order_fulfillment_status': 'mean', 'shipping_costs': 'mean'}).round(2)

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
print(f'\nImpact: 0 bn = {zero_delay:.2f}h, {max_bn} bn = {max_delay:.2f}h\n')

print('STATISTICAL PROOF\n')
single_corrs = []
for var in ['eta_variation_hours', 'loading_unloading_time', 'traffic_congestion_level']:
    corr = df[var].corr(df['shipping_costs'])
    single_corrs.append(corr)
    print(f'{var:30s} -> Cost: r={corr:+.4f}')

compound_corr = df['total_bn'].corr(df['shipping_costs'])
avg_single = np.mean([abs(c) for c in single_corrs])
print(f'\ntotal_bottlenecks -> Cost: r={compound_corr:+.4f}')
print(f'Compound is {abs(compound_corr)/avg_single:.1f}x stronger!\n')

clean_fulfill = df[df['total_bn'] == 0]['order_fulfillment_status'].mean() * 100
severe_fulfill = df[df['total_bn'] >= 4]['order_fulfillment_status'].mean() * 100
clean_cost = df[df['total_bn'] == 0]['shipping_costs'].mean()
severe_cost = df[df['total_bn'] >= 4]['shipping_costs'].mean()

print("CONCLUSION:")
print(f'Clean (0 bn): {clean_fulfill:.1f}% fulfill, cost')
print(f'Severe (4+ bn): {severe_fulfill:.1f}% fulfill, cost')
print(f'Impact: {clean_fulfill-severe_fulfill:.1f}pp drop, premium\n')
