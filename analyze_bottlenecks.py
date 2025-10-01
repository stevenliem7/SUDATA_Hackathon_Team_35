"""
Supply Chain Bottleneck Analysis
=================================
Purpose: Descriptive analysis of lead time, delivery performance, and costs
to identify operational bottlenecks
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BottleneckAnalyzer:
    """
    Analyze supply chain bottlenecks focusing on delivery performance and costs.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.analysis_report = {}
        
    def load_data(self):
        """Load the filtered dataset."""
        print("\nSupply Chain Bottleneck Analysis")
        print("Loading filtered dataset (Jan 2021 - Jan 2024)...\n")
        
        self.df = pd.read_csv(self.filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Loaded: {len(self.df):,} records")
        print(f"Period: {self.df['timestamp'].min().date()} to {self.df['timestamp'].max().date()}\n")
        
    def analyze_lead_time_performance(self):
        """Analyze lead time and delivery timing metrics."""
        print("=" * 80)
        print("LEAD TIME & DELIVERY PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Key metrics for lead time
        lead_time_cols = [
            'lead_time_days',
            'eta_variation_hours', 
            'delivery_time_deviation',
            'loading_unloading_time',
            'customs_clearance_time'
        ]
        
        print("\n1. LEAD TIME METRICS\n")
        
        # Lead Time Days Analysis
        print("Supplier Lead Time (days):")
        print(f"  Average: {self.df['lead_time_days'].mean():.2f} days")
        print(f"  Median: {self.df['lead_time_days'].median():.2f} days")
        print(f"  Std Dev: {self.df['lead_time_days'].std():.2f} days")
        print(f"  Min: {self.df['lead_time_days'].min():.2f} days")
        print(f"  Max: {self.df['lead_time_days'].max():.2f} days")
        print(f"  25th percentile: {self.df['lead_time_days'].quantile(0.25):.2f} days")
        print(f"  75th percentile: {self.df['lead_time_days'].quantile(0.75):.2f} days")
        
        # Identify long lead times (>75th percentile)
        long_lead_time_threshold = self.df['lead_time_days'].quantile(0.75)
        long_lead_times = (self.df['lead_time_days'] > long_lead_time_threshold).sum()
        print(f"\n  BOTTLENECK: {long_lead_times:,} records ({long_lead_times/len(self.df)*100:.1f}%) have lead times > {long_lead_time_threshold:.1f} days")
        
        print("\n" + "-" * 80)
        
        # ETA Variation Analysis
        print("\nETA Variation (hours):")
        print(f"  Average: {self.df['eta_variation_hours'].mean():.2f} hours")
        print(f"  Median: {self.df['eta_variation_hours'].median():.2f} hours")
        print(f"  Std Dev: {self.df['eta_variation_hours'].std():.2f} hours")
        print(f"  Min: {self.df['eta_variation_hours'].min():.2f} hours")
        print(f"  Max: {self.df['eta_variation_hours'].max():.2f} hours")
        
        # Late deliveries (positive ETA variation)
        late_deliveries = (self.df['eta_variation_hours'] > 0).sum()
        early_deliveries = (self.df['eta_variation_hours'] < 0).sum()
        on_time = (self.df['eta_variation_hours'] == 0).sum()
        
        print(f"\n  Late deliveries: {late_deliveries:,} ({late_deliveries/len(self.df)*100:.1f}%)")
        print(f"  Early deliveries: {early_deliveries:,} ({early_deliveries/len(self.df)*100:.1f}%)")
        print(f"  On-time: {on_time:,} ({on_time/len(self.df)*100:.1f}%)")
        
        # Severely late (>3 hours)
        severely_late = (self.df['eta_variation_hours'] > 3).sum()
        print(f"\n  BOTTLENECK: {severely_late:,} records ({severely_late/len(self.df)*100:.1f}%) are >3 hours late")
        
        print("\n" + "-" * 80)
        
        # Loading/Unloading Time Analysis
        print("\nLoading/Unloading Time (hours):")
        print(f"  Average: {self.df['loading_unloading_time'].mean():.2f} hours")
        print(f"  Median: {self.df['loading_unloading_time'].median():.2f} hours")
        print(f"  Std Dev: {self.df['loading_unloading_time'].std():.2f} hours")
        print(f"  Min: {self.df['loading_unloading_time'].min():.2f} hours")
        print(f"  Max: {self.df['loading_unloading_time'].max():.2f} hours")
        
        # Slow loading times (>75th percentile)
        slow_loading_threshold = self.df['loading_unloading_time'].quantile(0.75)
        slow_loading = (self.df['loading_unloading_time'] > slow_loading_threshold).sum()
        print(f"\n  BOTTLENECK: {slow_loading:,} records ({slow_loading/len(self.df)*100:.1f}%) have loading times > {slow_loading_threshold:.2f} hours")
        
        print("\n" + "-" * 80)
        
        # Customs Clearance Time Analysis
        print("\nCustoms Clearance Time (hours/days):")
        print(f"  Average: {self.df['customs_clearance_time'].mean():.2f}")
        print(f"  Median: {self.df['customs_clearance_time'].median():.2f}")
        print(f"  Std Dev: {self.df['customs_clearance_time'].std():.2f}")
        print(f"  Min: {self.df['customs_clearance_time'].min():.2f}")
        print(f"  Max: {self.df['customs_clearance_time'].max():.2f}")
        
        # Slow customs (>75th percentile)
        slow_customs_threshold = self.df['customs_clearance_time'].quantile(0.75)
        slow_customs = (self.df['customs_clearance_time'] > slow_customs_threshold).sum()
        print(f"\n  BOTTLENECK: {slow_customs:,} records ({slow_customs/len(self.df)*100:.1f}%) have customs times > {slow_customs_threshold:.2f}")
        
        print("\n" + "-" * 80)
        
        # Delivery Performance
        print("\nDelivery Time Deviation:")
        print(f"  Average: {self.df['delivery_time_deviation'].mean():.2f}")
        print(f"  Median: {self.df['delivery_time_deviation'].median():.2f}")
        print(f"  Std Dev: {self.df['delivery_time_deviation'].std():.2f}")
        
        # Store for reporting
        self.analysis_report['lead_time'] = {
            'avg_lead_time': self.df['lead_time_days'].mean(),
            'long_lead_times': long_lead_times,
            'late_deliveries': late_deliveries,
            'severely_late': severely_late,
            'slow_loading': slow_loading,
            'slow_customs': slow_customs
        }
        
    def analyze_delivery_performance_metrics(self):
        """Analyze order fulfillment and performance indicators."""
        print("\n" + "=" * 80)
        print("DELIVERY PERFORMANCE METRICS")
        print("=" * 80 + "\n")
        
        # Order Fulfillment Rate
        fulfillment_rate = self.df['order_fulfillment_status'].mean() * 100
        fulfilled_count = (self.df['order_fulfillment_status'] == 1).sum()
        not_fulfilled_count = (self.df['order_fulfillment_status'] == 0).sum()
        
        print("Order Fulfillment Status:")
        print(f"  Fulfillment Rate: {fulfillment_rate:.2f}%")
        print(f"  Fulfilled: {fulfilled_count:,} orders ({fulfilled_count/len(self.df)*100:.1f}%)")
        print(f"  Not Fulfilled: {not_fulfilled_count:,} orders ({not_fulfilled_count/len(self.df)*100:.1f}%)")
        print(f"\n  BOTTLENECK: {not_fulfilled_count:,} orders failed to fulfill on time")
        
        print("\n" + "-" * 80)
        
        # Delay Probability Analysis
        print("\nDelay Probability:")
        print(f"  Average: {self.df['delay_probability'].mean()*100:.2f}%")
        print(f"  Median: {self.df['delay_probability'].median()*100:.2f}%")
        print(f"  Std Dev: {self.df['delay_probability'].std()*100:.2f}%")
        
        # High delay risk
        high_delay_risk = (self.df['delay_probability'] > 0.7).sum()
        print(f"\n  High delay risk (>70%): {high_delay_risk:,} records ({high_delay_risk/len(self.df)*100:.1f}%)")
        
        # Correlation with actual fulfillment
        high_risk_fulfilled = self.df[self.df['delay_probability'] > 0.7]['order_fulfillment_status'].mean() * 100
        print(f"  Fulfillment rate for high-risk orders: {high_risk_fulfilled:.1f}%")
        
        print("\n" + "-" * 80)
        
        # Equipment Availability Impact
        print("\nHandling Equipment Availability:")
        equipment_availability = self.df['handling_equipment_availability'].mean() * 100
        print(f"  Average Availability: {equipment_availability:.2f}%")
        
        no_equipment = (self.df['handling_equipment_availability'] == 0).sum()
        print(f"  No Equipment Available: {no_equipment:,} instances ({no_equipment/len(self.df)*100:.1f}%)")
        
        # Fulfillment when equipment not available
        no_equip_fulfillment = self.df[self.df['handling_equipment_availability'] == 0]['order_fulfillment_status'].mean() * 100
        with_equip_fulfillment = self.df[self.df['handling_equipment_availability'] == 1]['order_fulfillment_status'].mean() * 100
        
        print(f"\n  Fulfillment with equipment: {with_equip_fulfillment:.1f}%")
        print(f"  Fulfillment without equipment: {no_equip_fulfillment:.1f}%")
        print(f"  BOTTLENECK IMPACT: {with_equip_fulfillment - no_equip_fulfillment:.1f}% fulfillment gap")
        
        self.analysis_report['performance'] = {
            'fulfillment_rate': fulfillment_rate,
            'not_fulfilled': not_fulfilled_count,
            'high_delay_risk': high_delay_risk,
            'no_equipment': no_equipment
        }
        
    def analyze_delivery_costs(self):
        """Analyze cost-related metrics."""
        print("\n" + "=" * 80)
        print("DELIVERY COST ANALYSIS")
        print("=" * 80 + "\n")
        
        # Shipping Costs
        print("Shipping Costs (USD):")
        print(f"  Total: ${self.df['shipping_costs'].sum():,.2f}")
        print(f"  Average per shipment: ${self.df['shipping_costs'].mean():.2f}")
        print(f"  Median: ${self.df['shipping_costs'].median():.2f}")
        print(f"  Std Dev: ${self.df['shipping_costs'].std():.2f}")
        print(f"  Min: ${self.df['shipping_costs'].min():.2f}")
        print(f"  Max: ${self.df['shipping_costs'].max():.2f}")
        
        # High-cost shipments
        high_cost_threshold = self.df['shipping_costs'].quantile(0.90)
        high_cost_shipments = (self.df['shipping_costs'] > high_cost_threshold).sum()
        high_cost_total = self.df[self.df['shipping_costs'] > high_cost_threshold]['shipping_costs'].sum()
        
        print(f"\n  High-cost shipments (>90th percentile, >${high_cost_threshold:.2f}):")
        print(f"    Count: {high_cost_shipments:,} ({high_cost_shipments/len(self.df)*100:.1f}%)")
        print(f"    Total cost: ${high_cost_total:,.2f} ({high_cost_total/self.df['shipping_costs'].sum()*100:.1f}% of total)")
        
        print("\n" + "-" * 80)
        
        # Fuel Consumption
        print("\nFuel Consumption (L/hour):")
        print(f"  Average: {self.df['fuel_consumption_rate'].mean():.2f} L/hour")
        print(f"  Median: {self.df['fuel_consumption_rate'].median():.2f} L/hour")
        print(f"  Std Dev: {self.df['fuel_consumption_rate'].std():.2f} L/hour")
        print(f"  Min: {self.df['fuel_consumption_rate'].min():.2f} L/hour")
        print(f"  Max: {self.df['fuel_consumption_rate'].max():.2f} L/hour")
        
        # High fuel consumption (>15 L/hour as per flags)
        high_fuel = (self.df['fuel_consumption_rate'] > 15).sum()
        print(f"\n  COST DRIVER: {high_fuel:,} records ({high_fuel/len(self.df)*100:.1f}%) have fuel consumption >15 L/hour")
        
        # Assuming $3 per liter fuel cost estimate
        fuel_cost_estimate = self.df['fuel_consumption_rate'].sum() * 3
        print(f"  Estimated total fuel cost: ${fuel_cost_estimate:,.2f} (assuming $3/liter)")
        
        print("\n" + "-" * 80)
        
        # Cost correlation with performance
        print("\nCost vs. Performance Correlation:")
        
        # Average cost for fulfilled vs not fulfilled
        fulfilled_avg_cost = self.df[self.df['order_fulfillment_status'] == 1]['shipping_costs'].mean()
        not_fulfilled_avg_cost = self.df[self.df['order_fulfillment_status'] == 0]['shipping_costs'].mean()
        
        print(f"  Average cost for fulfilled orders: ${fulfilled_avg_cost:.2f}")
        print(f"  Average cost for unfulfilled orders: ${not_fulfilled_avg_cost:.2f}")
        print(f"  Cost difference: ${abs(fulfilled_avg_cost - not_fulfilled_avg_cost):.2f}")
        
        # Cost by lead time quartiles
        self.df['lead_time_quartile'] = pd.qcut(self.df['lead_time_days'], q=4, labels=['Q1 (Fast)', 'Q2', 'Q3', 'Q4 (Slow)'])
        cost_by_lead_time = self.df.groupby('lead_time_quartile')['shipping_costs'].mean()
        
        print(f"\n  Average shipping cost by lead time quartile:")
        for quartile, cost in cost_by_lead_time.items():
            print(f"    {quartile}: ${cost:.2f}")
        
        self.analysis_report['costs'] = {
            'total_shipping': self.df['shipping_costs'].sum(),
            'avg_shipping': self.df['shipping_costs'].mean(),
            'high_cost_shipments': high_cost_shipments,
            'high_fuel_consumption': high_fuel
        }
        
    def analyze_bottleneck_correlations(self):
        """Identify correlations between bottleneck factors."""
        print("\n" + "=" * 80)
        print("BOTTLENECK CORRELATION ANALYSIS")
        print("=" * 80 + "\n")
        
        # Key bottleneck factors
        bottleneck_factors = [
            'lead_time_days',
            'loading_unloading_time',
            'customs_clearance_time',
            'traffic_congestion_level',
            'port_congestion_level',
            'route_risk_level',
            'delay_probability',
            'handling_equipment_availability',
            'weather_condition_severity'
        ]
        
        # Calculate correlation with order fulfillment
        print("Correlation with Order Fulfillment Status:")
        print("(Negative = reduces fulfillment, Positive = improves fulfillment)\n")
        
        correlations = []
        for factor in bottleneck_factors:
            if factor in self.df.columns:
                corr = self.df[factor].corr(self.df['order_fulfillment_status'])
                correlations.append((factor, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for factor, corr in correlations:
            impact = "NEGATIVE" if corr < 0 else "POSITIVE"
            strength = "STRONG" if abs(corr) > 0.3 else "MODERATE" if abs(corr) > 0.1 else "WEAK"
            print(f"  {factor:40s}: {corr:+.4f} ({strength} {impact})")
        
        print("\n" + "-" * 80)
        
        # Impact on shipping costs
        print("\nCorrelation with Shipping Costs:")
        print("(Positive = increases costs)\n")
        
        cost_correlations = []
        for factor in bottleneck_factors:
            if factor in self.df.columns:
                corr = self.df[factor].corr(self.df['shipping_costs'])
                cost_correlations.append((factor, corr))
        
        cost_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for factor, corr in cost_correlations:
            strength = "STRONG" if abs(corr) > 0.3 else "MODERATE" if abs(corr) > 0.1 else "WEAK"
            print(f"  {factor:40s}: {corr:+.4f} ({strength})")
        
    def identify_critical_bottlenecks(self):
        """Summarize the most critical bottlenecks."""
        print("\n" + "=" * 80)
        print("CRITICAL BOTTLENECKS SUMMARY")
        print("=" * 80 + "\n")
        
        bottlenecks = []
        
        # 1. Unfulfilled orders
        not_fulfilled_pct = (1 - self.df['order_fulfillment_status'].mean()) * 100
        bottlenecks.append(('Order Fulfillment', not_fulfilled_pct, 'orders not fulfilled on time'))
        
        # 2. Late deliveries
        late_pct = (self.df['eta_variation_hours'] > 0).sum() / len(self.df) * 100
        bottlenecks.append(('Late Deliveries', late_pct, 'deliveries arriving late'))
        
        # 3. Long lead times
        long_lead_pct = (self.df['lead_time_days'] > self.df['lead_time_days'].quantile(0.75)).sum() / len(self.df) * 100
        bottlenecks.append(('Long Lead Times', long_lead_pct, 'have extended lead times'))
        
        # 4. Slow loading/unloading
        slow_loading_pct = (self.df['loading_unloading_time'] > self.df['loading_unloading_time'].quantile(0.75)).sum() / len(self.df) * 100
        bottlenecks.append(('Slow Loading', slow_loading_pct, 'have slow loading/unloading'))
        
        # 5. Equipment unavailability
        no_equip_pct = (self.df['handling_equipment_availability'] == 0).sum() / len(self.df) * 100
        bottlenecks.append(('Equipment Shortage', no_equip_pct, 'lack handling equipment'))
        
        # 6. High traffic congestion
        high_traffic_pct = (self.df['traffic_congestion_level'] > 7).sum() / len(self.df) * 100
        bottlenecks.append(('Traffic Congestion', high_traffic_pct, 'face high traffic'))
        
        # 7. Customs delays
        slow_customs_pct = (self.df['customs_clearance_time'] > self.df['customs_clearance_time'].quantile(0.75)).sum() / len(self.df) * 100
        bottlenecks.append(('Customs Delays', slow_customs_pct, 'experience customs delays'))
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        print("Top Bottlenecks (by % of affected records):\n")
        for i, (name, pct, description) in enumerate(bottlenecks, 1):
            severity = "CRITICAL" if pct > 50 else "HIGH" if pct > 30 else "MODERATE"
            print(f"{i}. {name:25s} [{severity:8s}] {pct:5.1f}% {description}")
        
    def generate_recommendations(self):
        """Provide actionable recommendations."""
        print("\n" + "=" * 80)
        print("ACTIONABLE RECOMMENDATIONS")
        print("=" * 80 + "\n")
        
        recommendations = [
            ("1. IMPROVE FULFILLMENT RATE (Currently 61.6%)",
             [
                 "- Target: Increase to 80%+ industry standard",
                 "- Focus on the 38.4% unfulfilled orders",
                 "- Investigate root causes: equipment, traffic, customs"
             ]),
            
            ("2. REDUCE LATE DELIVERIES (50% arriving late)",
             [
                 "- Implement better route planning during high traffic periods",
                 "- Add buffer time for high-risk routes",
                 "- Review ETA prediction models for accuracy"
             ]),
            
            ("3. ADDRESS EQUIPMENT AVAILABILITY",
             [
                 f"- Equipment unavailable in {(self.df['handling_equipment_availability']==0).sum()/len(self.df)*100:.1f}% of cases",
                 f"- Fulfillment gap: {abs(self.df[self.df['handling_equipment_availability']==1]['order_fulfillment_status'].mean() - self.df[self.df['handling_equipment_availability']==0]['order_fulfillment_status'].mean())*100:.1f}% when equipment unavailable",
                 "- Invest in backup equipment or better scheduling"
             ]),
            
            ("4. OPTIMIZE LOADING/UNLOADING TIMES",
             [
                 f"- Average: {self.df['loading_unloading_time'].mean():.2f} hours",
                 "- 25% of loads take significantly longer",
                 "- Standardize processes and train staff"
             ]),
            
            ("5. STREAMLINE CUSTOMS CLEARANCE",
             [
                 f"- High variability in clearance times (std: {self.df['customs_clearance_time'].std():.2f})",
                 "- Pre-clear documentation for faster processing",
                 "- Work with customs brokers for high-volume routes"
             ]),
            
            ("6. MANAGE FUEL COSTS",
             [
                 f"- {(self.df['fuel_consumption_rate']>15).sum()} instances of high consumption (>15 L/hour)",
                 "- Route optimization to reduce fuel usage",
                 "- Vehicle maintenance and driver training"
             ]),
            
            ("7. SUPPLIER LEAD TIME MANAGEMENT",
             [
                 f"- Average lead time: {self.df['lead_time_days'].mean():.1f} days",
                 "- Work with suppliers to reduce variability",
                 "- Consider alternative suppliers for critical items"
             ])
        ]
        
        for title, points in recommendations:
            print(title)
            for point in points:
                print(point)
            print()
        
    def save_analysis_report(self):
        """Save detailed analysis to text file."""
        print("=" * 80)
        print("SAVING ANALYSIS REPORT")
        print("=" * 80 + "\n")
        
        report_path = 'bottleneck_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("SUPPLY CHAIN BOTTLENECK ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Analysis Period: Jan 2021 - Jan 2024\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Fulfillment Rate: {self.df['order_fulfillment_status'].mean()*100:.1f}%\n")
            f.write(f"Total Shipping Costs: ${self.df['shipping_costs'].sum():,.2f}\n")
            f.write(f"Average Lead Time: {self.df['lead_time_days'].mean():.1f} days\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write(f"- {(1-self.df['order_fulfillment_status'].mean())*100:.1f}% of orders not fulfilled on time\n")
            f.write(f"- {(self.df['eta_variation_hours']>0).sum()/len(self.df)*100:.1f}% of deliveries arriving late\n")
            f.write(f"- Equipment unavailable in {(self.df['handling_equipment_availability']==0).sum()/len(self.df)*100:.1f}% of cases\n")
            f.write(f"- {(self.df['fuel_consumption_rate']>15).sum()} instances of excessive fuel consumption\n\n")
            
            f.write("See console output for detailed analysis.\n")
        
        print(f"Analysis report saved: {report_path}\n")
        
    def run_complete_analysis(self):
        """Execute complete bottleneck analysis."""
        self.load_data()
        self.analyze_lead_time_performance()
        self.analyze_delivery_performance_metrics()
        self.analyze_delivery_costs()
        self.analyze_bottleneck_correlations()
        self.identify_critical_bottlenecks()
        self.generate_recommendations()
        self.save_analysis_report()
        
        print("Bottleneck analysis complete!\n")


def main():
    """Main execution function."""
    analyzer = BottleneckAnalyzer('filtered_supply_chain_data.csv')
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()

