"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine

from src.data.data_loader import DataLoader, get_engine, load_openfda_events
from src.config.config_manager import DataConfig


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock data configuration."""
        return DataConfig(
            warehouse_uri="postgresql+psycopg2://test:test@localhost:5432/test",
            tables={"openfda_events": "openfda_events", "synthea_events": "synthea_events"}
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return pd.DataFrame({
            'patientonsetage': [25, 30, 35, None, 40],
            'patientsex': ['M', 'F', 'M', 'F', 'M'],
            'reaction': ['nausea', 'headache', 'rash', 'fever', 'dizziness'],
            'brand_name': ['drug_a', 'drug_b', 'drug_a', 'drug_c', 'drug_b'],
            'serious': [1, 0, 1, 1, 0]
        })
    
    @patch('src.data.data_loader.create_engine')
    def test_init_success(self, mock_create_engine, mock_config):
        """Test successful DataLoader initialization."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DataLoader(mock_config)
        
        assert loader.config == mock_config
        assert loader.engine == mock_engine
        mock_create_engine.assert_called_once_with(mock_config.warehouse_uri)
    
    @patch('src.data.data_loader.create_engine')
    def test_init_connection_failure(self, mock_create_engine, mock_config):
        """Test DataLoader initialization with connection failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        with pytest.raises(Exception, match="Failed to connect to database"):
            DataLoader(mock_config)
    
    @patch('src.data.data_loader.create_engine')
    def test_load_data_success(self, mock_create_engine, mock_config, sample_data):
        """Test successful data loading."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch('pandas.read_sql', return_value=sample_data) as mock_read_sql:
            loader = DataLoader(mock_config)
            result = loader.load_data("openfda_events")
            
            assert result.equals(sample_data)
            mock_read_sql.assert_called_once()
    
    @patch('src.data.data_loader.create_engine')
    def test_load_data_invalid_table(self, mock_create_engine, mock_config):
        """Test data loading with invalid table name."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DataLoader(mock_config)
        
        with pytest.raises(ValueError, match="Table 'invalid_table' not configured"):
            loader.load_data("invalid_table")
    
    @patch('src.data.data_loader.create_engine')
    def test_load_data_with_filters(self, mock_create_engine, mock_config, sample_data):
        """Test data loading with filters."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch('pandas.read_sql', return_value=sample_data) as mock_read_sql:
            loader = DataLoader(mock_config)
            filters = {"serious": 1, "patientsex": "M"}
            result = loader.load_data("openfda_events", filters=filters)
            
            # Check that the query was built with filters
            call_args = mock_read_sql.call_args
            query = call_args[0][0]
            assert "WHERE" in query
            assert "serious = 1" in query
            assert "patientsex = 'M'" in query
    
    @patch('src.data.data_loader.create_engine')
    def test_load_data_with_columns(self, mock_create_engine, mock_config, sample_data):
        """Test data loading with specific columns."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch('pandas.read_sql', return_value=sample_data) as mock_read_sql:
            loader = DataLoader(mock_config)
            columns = ["patientonsetage", "serious"]
            result = loader.load_data("openfda_events", columns=columns)
            
            # Check that the query was built with specific columns
            call_args = mock_read_sql.call_args
            query = call_args[0][0]
            assert "SELECT patientonsetage, serious FROM" in query
    
    @patch('src.data.data_loader.create_engine')
    def test_load_data_with_limit(self, mock_create_engine, mock_config, sample_data):
        """Test data loading with limit."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch('pandas.read_sql', return_value=sample_data) as mock_read_sql:
            loader = DataLoader(mock_config)
            result = loader.load_data("openfda_events", limit=100)
            
            # Check that the query was built with limit
            call_args = mock_read_sql.call_args
            query = call_args[0][0]
            assert "LIMIT 100" in query
    
    @patch('src.data.data_loader.create_engine')
    def test_validate_data_quality_valid(self, mock_create_engine, mock_config, sample_data):
        """Test data quality validation with valid data."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DataLoader(mock_config)
        result = loader.validate_data_quality(sample_data, "serious")
        
        assert result['is_valid'] is True
        assert len(result['issues']) == 0
        assert result['statistics']['total_rows'] == 5
        assert result['statistics']['total_columns'] == 5
    
    @patch('src.data.data_loader.create_engine')
    def test_validate_data_quality_insufficient_rows(self, mock_create_engine, mock_config):
        """Test data quality validation with insufficient rows."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # Create data with insufficient rows
        small_data = pd.DataFrame({
            'serious': [1, 0]
        })
        
        loader = DataLoader(mock_config)
        result = loader.validate_data_quality(small_data, "serious", min_rows=100)
        
        assert result['is_valid'] is False
        assert "Insufficient data" in result['issues'][0]
    
    @patch('src.data.data_loader.create_engine')
    def test_validate_data_quality_missing_target(self, mock_create_engine, mock_config, sample_data):
        """Test data quality validation with missing target column."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DataLoader(mock_config)
        result = loader.validate_data_quality(sample_data, "missing_column")
        
        assert result['is_valid'] is False
        assert "Target column 'missing_column' not found" in result['issues'][0]
    
    @patch('src.data.data_loader.create_engine')
    def test_validate_data_quality_high_missing_ratio(self, mock_create_engine, mock_config):
        """Test data quality validation with high missing value ratio."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # Create data with high missing values
        data_with_missing = pd.DataFrame({
            'feature1': [1, 2, None, None, None],  # 60% missing
            'feature2': [1, 2, 3, 4, 5],
            'serious': [1, 0, 1, 0, 1]
        })
        
        loader = DataLoader(mock_config)
        result = loader.validate_data_quality(data_with_missing, "serious", max_missing_ratio=0.5)
        
        assert "High missing value ratios" in str(result['issues'])
    
    @patch('src.data.data_loader.create_engine')
    def test_get_data_info(self, mock_create_engine, mock_config):
        """Test getting data information."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # Mock count query result
        count_result = pd.DataFrame({'count': [1000]})
        
        # Mock column info result
        column_info = pd.DataFrame({
            'column_name': ['patientonsetage', 'serious'],
            'data_type': ['integer', 'integer'],
            'is_nullable': ['YES', 'NO']
        })
        
        # Mock sample data
        sample_data = pd.DataFrame({
            'patientonsetage': [25, 30],
            'serious': [1, 0]
        })
        
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.side_effect = [count_result, column_info, sample_data]
            
            loader = DataLoader(mock_config)
            result = loader.get_data_info("openfda_events")
            
            assert result['table_name'] == 'openfda_events'
            assert result['total_rows'] == 1000
            assert len(result['columns']) == 2
            assert len(result['sample_data']) == 2
    
    @patch('src.data.data_loader.create_engine')
    def test_close(self, mock_create_engine, mock_config):
        """Test closing database connection."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DataLoader(mock_config)
        loader.close()
        
        mock_engine.dispose.assert_called_once()


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @patch('src.data.data_loader.config_manager')
    def test_get_engine(self, mock_config_manager):
        """Test get_engine function."""
        mock_config = Mock()
        mock_config.data.warehouse_uri = "test_uri"
        mock_config_manager.load_config.return_value = mock_config
        
        with patch('src.data.data_loader.create_engine') as mock_create_engine:
            engine = get_engine()
            mock_create_engine.assert_called_once_with("test_uri")
    
    @patch('src.data.data_loader.DataLoader')
    def test_load_openfda_events(self, mock_data_loader_class):
        """Test load_openfda_events function."""
        mock_loader = Mock()
        mock_loader.load_data.return_value = pd.DataFrame({'test': [1, 2, 3]})
        mock_data_loader_class.return_value = mock_loader
        
        result = load_openfda_events()
        
        mock_loader.load_data.assert_called_once_with("openfda_events", None)
        mock_loader.close.assert_called_once()
        assert isinstance(result, pd.DataFrame)

