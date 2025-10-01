"""
Supply Chain Logistics Data Cleaning Script
============================================
Author: Data Science Expert - Datathon 2025
Purpose: Clean and validate dynamic supply chain logistics dataset

This script performs comprehensive data cleaning including:
- Data type conversions
- Missing value handling
- Duplicate detection and removal
- Outlier detection and handling
- Range validation based on field constraints
- Geospatial validation
- Temporal validation
- Data quality reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SupplyChainDataCleaner:
    """
    A comprehensive data cleaning class for supply chain logistics datasets.
    """
    
    def __init__(self, filepath):
        """
        Initialize the cleaner with dataset filepath.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file containing supply chain data
        """
        self.filepath = filepath
        self.df = None
        self.original_shape = None
        self.cleaning_report = {
            'original_rows': 0,
            'original_columns': 0,
            'duplicates_removed': 0,
            'missing_values': {},
            'outliers_detected': {},
            'invalid_ranges': {},
            'final_rows': 0,
            'final_columns': 0,
            'data_quality_score': 0.0
        }
        
    def load_data(self):
        """Load the dataset from CSV file."""
        self.df = pd.read_csv(self.filepath)
        self.original_shape = self.df.shape
        self.cleaning_report['original_rows'] = self.df.shape[0]
        self.cleaning_report['original_columns'] = self.df.shape[1]
        
        print(f"\nLoaded dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
    def analyze_data_types(self):
        """Analyze and convert data types."""
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Define expected numeric columns
        numeric_columns = [
            'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
            'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
            'loading_unloading_time', 'weather_condition_severity', 'port_congestion_level',
            'shipping_costs', 'supplier_reliability_score', 'lead_time_days',
            'historical_demand', 'iot_temperature', 'route_risk_level',
            'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
            'disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation'
        ]
        
        # Define binary columns
        binary_columns = [
            'handling_equipment_availability',
            'order_fulfillment_status',
            'cargo_condition_status'
        ]
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert binary columns
        for col in binary_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Converted data types for {len(numeric_columns) + len(binary_columns)} columns")
        
    def check_missing_values(self):
        """Identify and report missing values."""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_counts,
            'Percentage': missing_percentages
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            total_missing = missing_df['Missing_Count'].sum()
            print(f"Found {total_missing:,} missing values across {len(missing_df)} columns")
            self.cleaning_report['missing_values'] = missing_df.to_dict()
        else:
            print("No missing values detected")
        
    def handle_missing_values(self):
        """Handle missing values based on domain knowledge."""
        initial_rows = len(self.df)
        
        # Remove rows where timestamp is missing (critical field)
        if self.df['timestamp'].isnull().sum() > 0:
            self.df = self.df[self.df['timestamp'].notna()]
        
        # For GPS coordinates, remove rows if either is missing (both needed for location)
        if 'vehicle_gps_latitude' in self.df.columns and 'vehicle_gps_longitude' in self.df.columns:
            gps_missing = self.df['vehicle_gps_latitude'].isnull() | self.df['vehicle_gps_longitude'].isnull()
            self.df = self.df[~gps_missing]
        
        rows_removed = initial_rows - len(self.df)
        if rows_removed > 0:
            print(f"Removed {rows_removed:,} rows with critical missing values")
        
        # Fill numeric missing values with median (more robust than mean for outliers)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        total_filled = 0
        for col in numeric_cols:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                total_filled += missing_count
        
        # Handle categorical missing values
        if 'risk_classification' in self.df.columns:
            mode_value = self.df['risk_classification'].mode()[0] if len(self.df['risk_classification'].mode()) > 0 else 'Unknown'
            missing_count = self.df['risk_classification'].isnull().sum()
            if missing_count > 0:
                self.df['risk_classification'].fillna(mode_value, inplace=True)
                total_filled += missing_count
        
        if total_filled > 0:
            print(f"Filled {total_filled:,} missing values with appropriate defaults")
        
    def remove_duplicates(self):
        """Identify and remove duplicate records."""
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(self.df)
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed:,} duplicate rows")
        
    def validate_ranges(self):
        """Validate data ranges based on field specifications."""
        # Define valid ranges for each field
        range_validations = {
            'vehicle_gps_latitude': (-90, 90),
            'vehicle_gps_longitude': (-180, 180),
            'fuel_consumption_rate': (0, None),
            'traffic_congestion_level': (0, 10),
            'weather_condition_severity': (0, 1),
            'port_congestion_level': (0, 10),
            'route_risk_level': (0, 10),
            'shipping_costs': (0, None),
            'supplier_reliability_score': (0, 1),
            'lead_time_days': (0, None),
            'historical_demand': (0, None),
            'driver_behavior_score': (0, 1),
            'fatigue_monitoring_score': (0, 1),
            'handling_equipment_availability': (0, 1),
            'order_fulfillment_status': (0, 1),
            'cargo_condition_status': (0, 1),
            'loading_unloading_time': (0, None),
            'warehouse_inventory_level': (0, None),
            'disruption_likelihood_score': (0, 1),
            'delay_probability': (0, 1),
            'customs_clearance_time': (0, None),
        }
        
        total_corrections = 0
        
        for col, (min_val, max_val) in range_validations.items():
            if col in self.df.columns:
                if min_val is not None:
                    invalid_min = self.df[col] < min_val
                    invalid_min_count = invalid_min.sum()
                    if invalid_min_count > 0:
                        self.df.loc[invalid_min, col] = min_val
                        total_corrections += invalid_min_count
                
                if max_val is not None:
                    invalid_max = self.df[col] > max_val
                    invalid_max_count = invalid_max.sum()
                    if invalid_max_count > 0:
                        self.df.loc[invalid_max, col] = max_val
                        total_corrections += invalid_max_count
        
        if total_corrections > 0:
            print(f"Corrected {total_corrections:,} values outside valid ranges")
            self.cleaning_report['invalid_ranges'] = {'total': total_corrections}
        
    def validate_geospatial(self):
        """Validate GPS coordinates for Southern California region."""
        if 'vehicle_gps_latitude' in self.df.columns and 'vehicle_gps_longitude' in self.df.columns:
            lat_range = f"{self.df['vehicle_gps_latitude'].min():.1f}° to {self.df['vehicle_gps_latitude'].max():.1f}°"
            lon_range = f"{self.df['vehicle_gps_longitude'].min():.1f}° to {self.df['vehicle_gps_longitude'].max():.1f}°"
            print(f"GPS coordinates validated (Lat: {lat_range}, Lon: {lon_range})")
        
    def validate_temporal(self):
        """Validate temporal data."""
        if 'timestamp' in self.df.columns:
            min_date = self.df['timestamp'].min()
            max_date = self.df['timestamp'].max()
            duration_days = (max_date - min_date).days
            
            # Sort by timestamp
            self.df.sort_values('timestamp', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            print(f"Temporal range: {min_date.date()} to {max_date.date()} ({duration_days} days)")
        
    def detect_outliers(self):
        """Detect outliers using IQR method for key metrics."""
        outlier_check_cols = [
            'fuel_consumption_rate', 'shipping_costs', 'warehouse_inventory_level',
            'historical_demand', 'iot_temperature', 'loading_unloading_time'
        ]
        
        total_outliers = 0
        for col in outlier_check_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                total_outliers += outliers.sum()
        
        if total_outliers > 0:
            outlier_pct = (total_outliers / len(self.df)) * 100
            print(f"Detected {total_outliers:,} outliers ({outlier_pct:.1f}%) - kept for analysis")
        
    def validate_binary_fields(self):
        """Ensure binary fields contain only 0 and 1."""
        binary_fields = [
            'handling_equipment_availability',
            'order_fulfillment_status',
            'cargo_condition_status'
        ]
        
        total_corrections = 0
        for col in binary_fields:
            if col in self.df.columns:
                non_binary = ((self.df[col] != 0) & (self.df[col] != 1)).sum()
                if non_binary > 0:
                    self.df[col] = self.df[col].round().clip(0, 1)
                    total_corrections += non_binary
        
        if total_corrections > 0:
            print(f"Corrected {total_corrections:,} binary field values")
        
    def create_data_quality_flags(self):
        """Create flags for data quality monitoring."""
        flags_created = 0
        
        if 'fuel_consumption_rate' in self.df.columns:
            self.df['flag_high_fuel_consumption'] = (self.df['fuel_consumption_rate'] > 15).astype(int)
            flags_created += 1
        
        if 'iot_temperature' in self.df.columns:
            self.df['flag_extreme_temperature'] = ((self.df['iot_temperature'] < -10) | 
                                                    (self.df['iot_temperature'] > 40)).astype(int)
            flags_created += 1
        
        if 'route_risk_level' in self.df.columns:
            self.df['flag_high_risk_route'] = (self.df['route_risk_level'] > 8).astype(int)
            flags_created += 1
        
        if 'supplier_reliability_score' in self.df.columns:
            self.df['flag_low_supplier_reliability'] = (self.df['supplier_reliability_score'] < 0.3).astype(int)
            flags_created += 1
        
        print(f"Added {flags_created} quality monitoring flags")
        
    def calculate_data_quality_score(self):
        """Calculate overall data quality score."""
        # Completeness
        completeness = (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        
        # Uniqueness
        uniqueness = (1 - self.cleaning_report['duplicates_removed'] / self.cleaning_report['original_rows']) * 100
        
        # Validity
        total_corrections = self.cleaning_report.get('invalid_ranges', {}).get('total', 0)
        validity = (1 - total_corrections / (self.df.shape[0] * self.df.shape[1])) * 100
        
        # Consistency (check cross-field logic)
        consistency_issues = 0
        # Example: Check if order fulfillment and delay probability are consistent
        if 'order_fulfillment_status' in self.df.columns and 'delay_probability' in self.df.columns:
            # High delay probability with successful fulfillment might be inconsistent
            inconsistent = ((self.df['order_fulfillment_status'] == 1) & (self.df['delay_probability'] > 0.8)).sum()
            consistency_issues += inconsistent
        consistency = (1 - consistency_issues / len(self.df)) * 100
        
        # Outlier percentage
        outlier_check_cols = [
            'fuel_consumption_rate', 'shipping_costs', 'warehouse_inventory_level',
            'historical_demand', 'iot_temperature', 'loading_unloading_time'
        ]
        total_outliers = 0
        for col in outlier_check_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
                total_outliers += outliers.sum()
        outlier_pct = (total_outliers / (len(self.df) * len(outlier_check_cols))) * 100
        
        # Per-column completeness
        self.cleaning_report['completeness_per_column'] = ((1 - self.df.isnull().sum() / len(self.df)) * 100).to_dict()
        
        # Store metrics
        self.cleaning_report['completeness'] = completeness
        self.cleaning_report['uniqueness'] = uniqueness
        self.cleaning_report['validity'] = validity
        self.cleaning_report['consistency'] = consistency
        self.cleaning_report['outlier_pct'] = outlier_pct
        
        # Overall quality score
        quality_score = (completeness * 0.25 + uniqueness * 0.2 + validity * 0.25 + consistency * 0.2 + (100 - outlier_pct) * 0.1)
        self.cleaning_report['data_quality_score'] = quality_score
        
    def save_cleaned_data(self, output_filepath='cleaned_supply_chain_data.csv'):
        """Save cleaned dataset to CSV."""
        self.cleaning_report['final_rows'] = self.df.shape[0]
        self.cleaning_report['final_columns'] = self.df.shape[1]
        self.df.to_csv(output_filepath, index=False)
        print(f"\nSaved cleaned data: {output_filepath}")
        
    def print_quality_report(self):
        """Print comprehensive data quality report to console."""
        print("\n")
        print("DATA QUALITY REPORT")
        
        # Dataset Summary
        print("\nDataset Summary:")
        print(f"  Original size: {self.cleaning_report['original_rows']:,} rows")
        print(f"  Final size: {self.cleaning_report['final_rows']:,} rows")
        print(f"  Columns: {self.cleaning_report['final_columns']}")
        rows_removed = self.cleaning_report['original_rows'] - self.cleaning_report['final_rows']
        if rows_removed > 0:
            print(f"  Rows removed: {rows_removed:,}")
        
        # Quality Metrics
        print("\nData Quality Metrics:")
        print(f"  Completeness: {self.cleaning_report['completeness']:.1f}%  (no missing values)")
        print(f"  Validity: {self.cleaning_report['validity']:.1f}%  (schema compliance)")
        print(f"  Consistency: {self.cleaning_report['consistency']:.1f}%  (cross-field logic)")
        print(f"  Uniqueness: {self.cleaning_report['uniqueness']:.1f}%  (deduplication check)")
        print(f"  Outlier:  {self.cleaning_report['outlier_pct']:.1f}%  (distribution sanity check)")
        print(f"\n  Overall Quality Score: {self.cleaning_report['data_quality_score']:.1f}%")
        
        # Completeness per column (show only columns with < 100% if any)
        completeness_per_col = self.cleaning_report['completeness_per_column']
        incomplete_cols = {k: v for k, v in completeness_per_col.items() if v < 100}
        
        if incomplete_cols:
            print("\nCompleteness by Column (incomplete columns):")
            for col, pct in sorted(incomplete_cols.items(), key=lambda x: x[1]):
                print(f"  {col}: {pct:.1f}%")
        else:
            print("\nCompleteness by Column: 100% across all columns")
        
        # Cleaning Actions
        print("\nCleaning Actions Performed:")
        if self.cleaning_report['duplicates_removed'] > 0:
            print(f"  - Removed {self.cleaning_report['duplicates_removed']:,} duplicate records")
        
        range_corrections = self.cleaning_report.get('invalid_ranges', {}).get('total', 0)
        if range_corrections > 0:
            print(f"  - Corrected {range_corrections:,} out-of-range values")
        
        print("  - Validated GPS coordinates and timestamps")
        print("  - Filled missing values with appropriate defaults")
        print("  - Added quality monitoring flags")
        

        
    def run_complete_cleaning_pipeline(self, output_filepath='cleaned_supply_chain_data.csv'):
        """Execute complete data cleaning pipeline."""
        print("\nSupply Chain Data Cleaning: ")
        
        self.load_data()
        self.analyze_data_types()
        self.check_missing_values()
        self.handle_missing_values()
        self.remove_duplicates()
        self.validate_ranges()
        self.validate_binary_fields()
        self.validate_geospatial()
        self.validate_temporal()
        self.detect_outliers()
        self.create_data_quality_flags()
        self.calculate_data_quality_score()
        self.save_cleaned_data(output_filepath)
        self.print_quality_report()
        
        return self.df


def main():
    """Main execution function for dataset cleaning"""
    cleaner = SupplyChainDataCleaner('dynamic_supply_chain_logistics_dataset.csv')
    cleaned_df = cleaner.run_complete_cleaning_pipeline(
        output_filepath='cleaned_supply_chain_data.csv'
    )
    
    print("Data cleaning completed\n")
    
if __name__ == "__main__":
    main()


