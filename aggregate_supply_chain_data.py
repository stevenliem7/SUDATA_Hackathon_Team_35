"""
Purpose: Filter and aggregatedata to declared range and create time-series aggregation.

This script:
- Filters data to declared temporal range (Jan 2021 - Jan 2024)
- Highlights data beyond declared range
- Aggregates by day and week with key performance metrics for analysis purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SupplyChainAggregator:
    """
    Filter and aggregate supply chain data for trend analysis.
    """
    
    def __init__(self, filepath):
        """
        Initialize with cleaned dataset filepath.
        
        Parameters:
        -----------
        filepath : str
            Path to cleaned CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_filtered = None
        self.df_daily = None
        self.df_weekly = None
        self.temporal_report = {}
        
    def load_data(self):
        """Load the cleaned dataset."""
        print("\nSupply Chain Data Aggregation - Datathon 2025")
        print("Loading cleaned dataset...\n")
        
        self.df = pd.read_csv(self.filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Loaded dataset: {len(self.df):,} rows")
        print(f"Date range: {self.df['timestamp'].min().date()} to {self.df['timestamp'].max().date()}")
        
    def analyze_temporal_discrepancy(self):
        """Analyze and report temporal range discrepancy."""
        print("\nTEMPORAL RANGE ANALYSIS")
        
        # Declared range
        declared_start = pd.Timestamp('2021-01-01')
        declared_end = pd.Timestamp('2024-01-31')  # End of January 2024
        
        # Actual range
        actual_start = self.df['timestamp'].min()
        actual_end = self.df['timestamp'].max()
        
        # Count records in different periods
        within_declared = self.df[
            (self.df['timestamp'] >= declared_start) & 
            (self.df['timestamp'] <= declared_end)
        ]
        
        beyond_declared = self.df[self.df['timestamp'] > declared_end]
        
        print(f"\nDeclared Range: {declared_start.date()} to {declared_end.date()}")
        print(f"Actual Range:   {actual_start.date()} to {actual_end.date()}")
        
        if len(beyond_declared) > 0:
            months_beyond = (actual_end.year - declared_end.year) * 12 + (actual_end.month - declared_end.month)
            print(f"\nTEMPORAL DISCREPANCY DETECTED:")
            print(f"  Extension: {months_beyond} months beyond declared range")
            print(f"  Records within declared range: {len(within_declared):,} ({len(within_declared)/len(self.df)*100:.1f}%)")
            print(f"  Records beyond declared range: {len(beyond_declared):,} ({len(beyond_declared)/len(self.df)*100:.1f}%)")
            print(f"  Extended period: {declared_end.date()} to {actual_end.date()}")
        
        self.temporal_report = {
            'declared_start': declared_start,
            'declared_end': declared_end,
            'actual_start': actual_start,
            'actual_end': actual_end,
            'within_count': len(within_declared),
            'beyond_count': len(beyond_declared)
        }
        
    def filter_to_declared_range(self):
        """Filter dataset to declared temporal range only."""
        print("\nFILTERING TO DECLARED RANGE")
        
        declared_start = pd.Timestamp('2021-01-01')
        declared_end = pd.Timestamp('2024-01-31')
        
        self.df_filtered = self.df[
            (self.df['timestamp'] >= declared_start) & 
            (self.df['timestamp'] <= declared_end)
        ].copy()
        
        rows_removed = len(self.df) - len(self.df_filtered)
        
        print(f"Filtered dataset: {len(self.df_filtered):,} rows")
        print(f"Removed {rows_removed:,} rows outside declared range")
        print(f"Date range: {self.df_filtered['timestamp'].min().date()} to {self.df_filtered['timestamp'].max().date()}")
        
    def aggregate_daily(self):
        """Aggregate data by day with key performance metrics."""
        print("\nAGGREGATING DAILY METRICS")
        
        # Set timestamp as index for resampling
        df_temp = self.df_filtered.set_index('timestamp')
        
        # Aggregate by day
        daily_agg = {
            # Volume metrics
            'vehicle_gps_latitude': 'count',  # Number of records per day
            
            # Operational metrics
            'fuel_consumption_rate': ['mean', 'std', 'max'],
            'shipping_costs': ['mean', 'sum', 'std'],
            'loading_unloading_time': ['mean', 'max'],
            'warehouse_inventory_level': ['mean', 'min', 'max'],
            'historical_demand': ['mean', 'sum'],
            
            # Performance metrics
            'order_fulfillment_status': 'mean',  # Fulfillment rate
            'eta_variation_hours': ['mean', 'std'],
            'delivery_time_deviation': ['mean', 'std'],
            
            # Risk metrics
            'traffic_congestion_level': 'mean',
            'route_risk_level': 'mean',
            'delay_probability': 'mean',
            'disruption_likelihood_score': 'mean',
            
            # Quality metrics
            'cargo_condition_status': 'mean',  # Good condition rate
            'supplier_reliability_score': 'mean',
            'driver_behavior_score': 'mean',
            'fatigue_monitoring_score': 'mean',
            
            # Environmental metrics
            'iot_temperature': ['mean', 'min', 'max'],
            'weather_condition_severity': 'mean',
            
            # Equipment availability
            'handling_equipment_availability': 'mean',
        }
        
        self.df_daily = df_temp.resample('D').agg(daily_agg)
        
        # Flatten column names
        self.df_daily.columns = ['_'.join(col).strip('_') for col in self.df_daily.columns.values]
        
        # Rename for clarity
        self.df_daily.rename(columns={
            'vehicle_gps_latitude_count': 'daily_record_count',
            'order_fulfillment_status_mean': 'fulfillment_rate',
            'cargo_condition_status_mean': 'good_cargo_rate',
            'handling_equipment_availability_mean': 'equipment_availability_rate'
        }, inplace=True)
        
        # Remove days with no data
        self.df_daily = self.df_daily[self.df_daily['daily_record_count'] > 0]
        
        # Reset index to make date a column
        self.df_daily.reset_index(inplace=True)
        self.df_daily.rename(columns={'timestamp': 'date'}, inplace=True)
        
        print(f"Daily aggregation complete: {len(self.df_daily):,} days")
        print(f"Columns: {len(self.df_daily.columns)}")
        
    def aggregate_weekly(self):
        """Aggregate data by week with key performance metrics."""
        print("\nAGGREGATING WEEKLY METRICS")
        
        # Set timestamp as index for resampling
        df_temp = self.df_filtered.set_index('timestamp')
        
        # Aggregate by week (Monday-Sunday)
        weekly_agg = {
            # Volume metrics
            'vehicle_gps_latitude': 'count',  # Number of records per week
            
            # Operational metrics
            'fuel_consumption_rate': ['mean', 'std', 'max'],
            'shipping_costs': ['mean', 'sum', 'std'],
            'loading_unloading_time': ['mean', 'max'],
            'warehouse_inventory_level': ['mean', 'min', 'max'],
            'historical_demand': ['mean', 'sum'],
            
            # Performance metrics
            'order_fulfillment_status': 'mean',  # Fulfillment rate
            'eta_variation_hours': ['mean', 'std'],
            'delivery_time_deviation': ['mean', 'std'],
            
            # Risk metrics
            'traffic_congestion_level': 'mean',
            'route_risk_level': 'mean',
            'delay_probability': 'mean',
            'disruption_likelihood_score': 'mean',
            
            # Quality metrics
            'cargo_condition_status': 'mean',  # Good condition rate
            'supplier_reliability_score': 'mean',
            'driver_behavior_score': 'mean',
            'fatigue_monitoring_score': 'mean',
            
            # Environmental metrics
            'iot_temperature': ['mean', 'min', 'max'],
            'weather_condition_severity': 'mean',
            
            # Equipment availability
            'handling_equipment_availability': 'mean',
            
            # Port and customs
            'port_congestion_level': 'mean',
            'customs_clearance_time': 'mean',
        }
        
        self.df_weekly = df_temp.resample('W-MON').agg(weekly_agg)
        
        # Flatten column names
        self.df_weekly.columns = ['_'.join(col).strip('_') for col in self.df_weekly.columns.values]
        
        # Rename for clarity
        self.df_weekly.rename(columns={
            'vehicle_gps_latitude_count': 'weekly_record_count',
            'order_fulfillment_status_mean': 'fulfillment_rate',
            'cargo_condition_status_mean': 'good_cargo_rate',
            'handling_equipment_availability_mean': 'equipment_availability_rate'
        }, inplace=True)
        
        # Remove weeks with no data
        self.df_weekly = self.df_weekly[self.df_weekly['weekly_record_count'] > 0]
        
        # Reset index to make date a column
        self.df_weekly.reset_index(inplace=True)
        self.df_weekly.rename(columns={'timestamp': 'week_start'}, inplace=True)
        
        print(f"Weekly aggregation complete: {len(self.df_weekly):,} weeks")
        print(f"Columns: {len(self.df_weekly.columns)}")
        
    def save_datasets(self):
        """Save all processed datasets."""
        print("\nSAVING DATASETS")
        
        # Save filtered dataset (declared range only)
        filtered_path = 'filtered_supply_chain_data.csv'
        self.df_filtered.to_csv(filtered_path, index=False)
        print(f"Filtered dataset: {filtered_path}")
        print(f"  {len(self.df_filtered):,} rows, Jan 2021 - Jan 2024 only")
        
        # Save daily aggregation
        daily_path = 'daily_supply_chain_metrics.csv'
        self.df_daily.to_csv(daily_path, index=False)
        print(f"\nDaily aggregation: {daily_path}")
        print(f"  {len(self.df_daily):,} days")
        print(f"  {len(self.df_daily.columns)} metrics per day")
        
        # Save weekly aggregation
        weekly_path = 'weekly_supply_chain_metrics.csv'
        self.df_weekly.to_csv(weekly_path, index=False)
        print(f"\nWeekly aggregation: {weekly_path}")
        print(f"  {len(self.df_weekly):,} weeks")
        print(f"  {len(self.df_weekly.columns)} metrics per week")
        
    def print_summary(self):
        """Print summary of key metrics."""
        print("\nKEY METRICS SUMMARY (Declared Range: Jan 2021 - Jan 2024)")
        
        print("\nOperational Performance:")
        print(f"  Average Fulfillment Rate: {self.df_filtered['order_fulfillment_status'].mean()*100:.1f}%")
        print(f"  Average Fuel Consumption: {self.df_filtered['fuel_consumption_rate'].mean():.2f} L/hour")
        print(f"  Total Shipping Costs: ${self.df_filtered['shipping_costs'].sum():,.2f}")
        print(f"  Average Daily Demand: {self.df_filtered['historical_demand'].mean():,.0f} units")
        
        print("\nRisk Indicators:")
        print(f"  Average Traffic Congestion: {self.df_filtered['traffic_congestion_level'].mean():.1f}/10")
        print(f"  Average Route Risk: {self.df_filtered['route_risk_level'].mean():.1f}/10")
        print(f"  Average Delay Probability: {self.df_filtered['delay_probability'].mean()*100:.1f}%")
        
        print("\nQuality Metrics:")
        print(f"  Good Cargo Condition Rate: {self.df_filtered['cargo_condition_status'].mean()*100:.1f}%")
        print(f"  Average Supplier Reliability: {self.df_filtered['supplier_reliability_score'].mean()*100:.1f}%")
        print(f"  Average Driver Behavior Score: {self.df_filtered['driver_behavior_score'].mean()*100:.1f}%")
        
        # Trend insights from weekly data
        if len(self.df_weekly) > 4:
            print("\nTrend Analysis (First 4 weeks vs. Last 4 weeks):")
            
            first_4_weeks = self.df_weekly.head(4)
            last_4_weeks = self.df_weekly.tail(4)
            
            fulfillment_change = (last_4_weeks['fulfillment_rate'].mean() - first_4_weeks['fulfillment_rate'].mean()) * 100
            fuel_change = ((last_4_weeks['fuel_consumption_rate_mean'].mean() - first_4_weeks['fuel_consumption_rate_mean'].mean()) / first_4_weeks['fuel_consumption_rate_mean'].mean()) * 100
            
            print(f"  Fulfillment Rate Change: {fulfillment_change:+.1f} percentage points")
            print(f"  Fuel Consumption Change: {fuel_change:+.1f}%")
        
    def run_complete_pipeline(self):
        """Execute complete filtering and aggregation pipeline."""
        self.load_data()
        self.analyze_temporal_discrepancy()
        self.filter_to_declared_range()
        self.aggregate_daily()
        self.aggregate_weekly()
        self.save_datasets()
        self.print_summary()
        
        print("\nAggregation complete! Ready for time-series analysis.\n")
        

def main():
    """Main execution function."""
    aggregator = SupplyChainAggregator('cleaned_supply_chain_data.csv')
    aggregator.run_complete_pipeline()
    

if __name__ == "__main__":
    main()

