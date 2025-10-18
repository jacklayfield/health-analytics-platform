"""
Enhanced data loading utilities with validation and error handling.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from ..config.config_manager import config_manager, DataConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Enhanced data loader with validation and error handling."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration. If None, loads from config manager.
        """
        if config is None:
            config = config_manager.load_config().data
        
        self.config = config
        self.engine = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.engine = create_engine(self.config.warehouse_uri)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data from specified table with optional filtering.
        
        Args:
            table_name: Name of the table to load from
            columns: List of columns to select (None for all)
            filters: Dictionary of column filters
            limit: Maximum number of rows to return
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If table_name is not configured
            RuntimeError: If data loading fails
        """
        if table_name not in self.config.tables:
            raise ValueError(f"Table '{table_name}' not configured. Available: {list(self.config.tables.keys())}")
        
        actual_table = self.config.tables[table_name]
        
        try:
            # Build query
            query = self._build_query(actual_table, columns, filters, limit)
            
            logger.info(f"Loading data from {actual_table}")
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                logger.warning(f"Query returned no rows from {actual_table}")
            else:
                logger.info(f"Loaded {len(df)} rows from {actual_table}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {actual_table}: {e}")
            raise RuntimeError(f"Data loading failed: {e}")
    
    def _build_query(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> str:
        """Build SQL query with optional filters and limits."""
        # Select clause
        if columns:
            col_clause = ", ".join(columns)
        else:
            col_clause = "*"
        
        query = f"SELECT {col_clause} FROM {table}"
        
        # Where clause
        if filters:
            conditions = []
            for col, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"{col} = '{value}'")
                elif isinstance(value, (int, float)):
                    conditions.append(f"{col} = {value}")
                elif isinstance(value, list):
                    if all(isinstance(v, str) for v in value):
                        values_str = "', '".join(value)
                        conditions.append(f"{col} IN ('{values_str}')")
                    else:
                        values_str = ", ".join(map(str, value))
                        conditions.append(f"{col} IN ({values_str})")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        # Limit clause
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    def get_data_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about the data in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with data information
        """
        if table_name not in self.config.tables:
            raise ValueError(f"Table '{table_name}' not configured")
        
        actual_table = self.config.tables[table_name]
        
        try:
            # Get basic info
            count_query = f"SELECT COUNT(*) as count FROM {actual_table}"
            count_result = pd.read_sql(count_query, self.engine)
            total_rows = count_result['count'].iloc[0]
            
            # Get column info
            info_query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{actual_table}'
            ORDER BY ordinal_position
            """
            column_info = pd.read_sql(info_query, self.engine)
            
            # Get sample data
            sample_query = f"SELECT * FROM {actual_table} LIMIT 5"
            sample_data = pd.read_sql(sample_query, self.engine)
            
            return {
                'table_name': actual_table,
                'total_rows': total_rows,
                'columns': column_info.to_dict('records'),
                'sample_data': sample_data.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Failed to get data info for {actual_table}: {e}")
            raise
    
    def validate_data_quality(
        self,
        df: pd.DataFrame,
        target_column: str,
        min_rows: int = 100,
        max_missing_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate data quality for ML pipeline.
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum allowed ratio of missing values
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check minimum rows
        if len(df) < min_rows:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Insufficient data: {len(df)} rows (minimum: {min_rows})")
        
        # Check target column
        if target_column not in df.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Target column '{target_column}' not found")
        else:
            # Check target distribution
            target_dist = df[target_column].value_counts()
            validation_results['statistics']['target_distribution'] = target_dist.to_dict()
            
            # Check for sufficient class balance
            min_class_count = target_dist.min()
            if min_class_count < 10:
                validation_results['issues'].append(f"Very small class size: {min_class_count}")
        
        # Check missing values
        missing_ratios = df.isnull().sum() / len(df)
        high_missing = missing_ratios[missing_ratios > max_missing_ratio]
        
        if not high_missing.empty:
            validation_results['issues'].append(
                f"High missing value ratios: {high_missing.to_dict()}"
            )
        
        validation_results['statistics']['missing_ratios'] = missing_ratios.to_dict()
        validation_results['statistics']['total_rows'] = len(df)
        validation_results['statistics']['total_columns'] = len(df.columns)
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        if validation_results['issues']:
            logger.warning(f"Data quality issues found: {validation_results['issues']}")
        
        return validation_results
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Convenience functions for backward compatibility
def get_engine():
    """Get database engine (backward compatibility)."""
    config = config_manager.load_config().data
    return create_engine(config.warehouse_uri)


def load_openfda_events(columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load OpenFDA events data (backward compatibility)."""
    loader = DataLoader()
    try:
        return loader.load_data("openfda_events", columns)
    finally:
        loader.close()

