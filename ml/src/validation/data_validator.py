"""
Data validation framework using Great Expectations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite
    from great_expectations.dataset import PandasDataset
    from great_expectations.data_context import DataContext
    from great_expectations.data_context.types.base import DataContextConfig
    from great_expectations.core.batch import RuntimeBatchRequest
except ImportError:
    ge = None
    ExpectationSuite = None
    PandasDataset = None
    DataContext = None
    DataContextConfig = None
    RuntimeBatchRequest = None

from ..config.config_manager import config_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Data validation framework using Great Expectations."""
    
    def __init__(self, expectations_dir: Optional[str] = None):
        """
        Initialize data validator.
        
        Args:
            expectations_dir: Directory to store expectation suites
        """
        if ge is None:
            raise ImportError(
                "Great Expectations is not installed. "
                "Install it with: pip install great-expectations"
            )
        
        self.expectations_dir = Path(expectations_dir) if expectations_dir else Path("expectations")
        self.expectations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Great Expectations context
        self.context = self._create_context()
        
        logger.info(f"Data validator initialized with expectations directory: {self.expectations_dir}")
    
    def _create_context(self) -> DataContext:
        """Create Great Expectations data context."""
        try:
            # Try to load existing context
            context = DataContext()
            logger.info("Loaded existing Great Expectations context")
            return context
        except Exception:
            # Create new context
            config = DataContextConfig(
                config_version=3.0,
                datasources={
                    "pandas_datasource": {
                        "class_name": "Datasource",
                        "execution_engine": {
                            "class_name": "PandasExecutionEngine"
                        },
                        "data_connectors": {
                            "runtime_data_connector": {
                                "class_name": "RuntimeDataConnector",
                                "batch_identifiers": ["default_identifier_name"]
                            }
                        }
                    }
                },
                stores={
                    "expectations_store": {
                        "class_name": "ExpectationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(self.expectations_dir / "expectations")
                        }
                    },
                    "validations_store": {
                        "class_name": "ValidationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(self.expectations_dir / "validations")
                        }
                    },
                    "evaluation_parameter_store": {
                        "class_name": "EvaluationParameterStore"
                    }
                },
                expectations_store_name="expectations_store",
                validations_store_name="validations_store",
                evaluation_parameter_store_name="evaluation_parameter_store",
                data_docs_sites={
                    "local_site": {
                        "class_name": "SiteBuilder",
                        "show_how_to_buttons": True,
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(self.expectations_dir / "data_docs")
                        },
                        "site_index_builder": {
                            "class_name": "DefaultSiteIndexBuilder"
                        }
                    }
                },
                validation_operators={
                    "action_list_operator": {
                        "class_name": "ActionListValidationOperator",
                        "action_list": [
                            {
                                "name": "store_validation_result",
                                "action": {
                                    "class_name": "StoreValidationResultAction"
                                }
                            },
                            {
                                "name": "store_evaluation_params",
                                "action": {
                                    "class_name": "StoreEvaluationParametersAction"
                                }
                            },
                            {
                                "name": "update_data_docs",
                                "action": {
                                    "class_name": "UpdateDataDocsAction"
                                }
                            }
                        ]
                    }
                }
            )
            
            context = DataContext(config)
            logger.info("Created new Great Expectations context")
            return context
    
    def create_expectation_suite(
        self,
        suite_name: str,
        data_type: str = "openfda_events"
    ) -> ExpectationSuite:
        """
        Create expectation suite for a specific data type.
        
        Args:
            suite_name: Name of the expectation suite
            data_type: Type of data (e.g., 'openfda_events')
            
        Returns:
            Created expectation suite
        """
        try:
            suite = self.context.create_expectation_suite(
                expectation_suite_name=suite_name,
                overwrite_existing=True
            )
            
            # Add expectations based on data type
            if data_type == "openfda_events":
                self._add_openfda_expectations(suite)
            elif data_type == "synthea_events":
                self._add_synthea_expectations(suite)
            else:
                self._add_generic_expectations(suite)
            
            # Save suite
            self.context.save_expectation_suite(suite)
            logger.info(f"Created expectation suite: {suite_name}")
            
            return suite
            
        except Exception as e:
            logger.error(f"Failed to create expectation suite {suite_name}: {e}")
            raise
    
    def _add_openfda_expectations(self, suite: ExpectationSuite) -> None:
        """Add expectations specific to OpenFDA events data."""
        # Basic data quality expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("patientonsetage")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("patientsex")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("reaction")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("brand_name")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("serious")
        )
        
        # Patient age expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToBeBetween(
                column="patientonsetage",
                min_value=0,
                max_value=120,
                mostly=0.95  # Allow 5% outliers
            )
        )
        
        # Patient sex expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="patientsex",
                value_set=["M", "F", "U", "Unknown"]
            )
        )
        
        # Serious field expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToBeInSet(
                column="serious",
                value_set=[0, 1, 2]
            )
        )
        
        # Missing value expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToNotBeNull(
                column="serious",
                mostly=0.8  # Allow 20% missing
            )
        )
        
        # Data completeness expectations
        suite.add_expectation(
            ge.expectations.ExpectTableRowCountToBeBetween(
                min_value=1000,
                max_value=1000000
            )
        )
    
    def _add_synthea_expectations(self, suite: ExpectationSuite) -> None:
        """Add expectations specific to Synthea events data."""
        # Add Synthea-specific expectations here
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("patient_id")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("encounter_id")
        )
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist("condition_code")
        )
    
    def _add_generic_expectations(self, suite: ExpectationSuite) -> None:
        """Add generic data quality expectations."""
        suite.add_expectation(
            ge.expectations.ExpectTableRowCountToBeBetween(
                min_value=1,
                max_value=10000000
            )
        )
        
        suite.add_expectation(
            ge.expectations.ExpectTableColumnCountToBeBetween(
                min_value=1,
                max_value=1000
            )
        )
    
    def validate_data(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate data against expectation suite.
        
        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            data_asset_name: Name for the data asset
            
        Returns:
            Validation results
        """
        try:
            if data_asset_name is None:
                data_asset_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name=data_asset_name,
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Run validation
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            validation_result = validator.validate()
            
            # Process results
            results = self._process_validation_results(validation_result)
            
            logger.info(f"Data validation completed for {suite_name}")
            logger.info(f"Success rate: {results['success_rate']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Data validation failed for {suite_name}: {e}")
            raise
    
    def _process_validation_results(self, validation_result) -> Dict[str, Any]:
        """Process validation results into a structured format."""
        results = {
            'success': validation_result.success,
            'success_rate': validation_result.statistics['successful_expectations'] / 
                           validation_result.statistics['evaluated_expectations'],
            'total_expectations': validation_result.statistics['evaluated_expectations'],
            'successful_expectations': validation_result.statistics['successful_expectations'],
            'failed_expectations': validation_result.statistics['unsuccessful_expectations'],
            'expectation_results': [],
            'failed_expectations_details': []
        }
        
        # Process individual expectation results
        for result in validation_result.results:
            expectation_result = {
                'expectation_type': result.expectation_config.expectation_type,
                'column': result.expectation_config.kwargs.get('column'),
                'success': result.success,
                'result': result.result
            }
            
            results['expectation_results'].append(expectation_result)
            
            if not result.success:
                results['failed_expectations_details'].append({
                    'expectation_type': result.expectation_config.expectation_type,
                    'column': result.expectation_config.kwargs.get('column'),
                    'kwargs': result.expectation_config.kwargs,
                    'result': result.result
                })
        
        return results
    
    def validate_training_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        data_type: str = "openfda_events"
    ) -> Dict[str, Any]:
        """
        Validate training data with ML-specific checks.
        
        Args:
            df: Training DataFrame
            target_column: Name of target column
            data_type: Type of data
            
        Returns:
            Validation results
        """
        suite_name = f"{data_type}_training_validation"
        
        # Create or get expectation suite
        try:
            suite = self.context.get_expectation_suite(suite_name)
        except Exception:
            suite = self.create_expectation_suite(suite_name, data_type)
        
        # Add ML-specific expectations
        self._add_ml_expectations(suite, target_column)
        self.context.save_expectation_suite(suite)
        
        # Run validation
        return self.validate_data(df, suite_name)
    
    def _add_ml_expectations(self, suite: ExpectationSuite, target_column: str) -> None:
        """Add ML-specific expectations."""
        # Target column expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist(target_column)
        )
        
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToNotBeNull(
                column=target_column,
                mostly=0.9  # Allow 10% missing targets
            )
        )
        
        # Class balance expectations
        suite.add_expectation(
            ge.expectations.ExpectColumnValueCountsToBeInSet(
                column=target_column,
                value_set=[0, 1, 2]  # Adjust based on your target classes
            )
        )
        
        # Data drift expectations (basic)
        suite.add_expectation(
            ge.expectations.ExpectTableRowCountToBeBetween(
                min_value=1000,  # Minimum samples for training
                max_value=10000000
            )
        )
    
    def generate_data_quality_report(
        self,
        df: pd.DataFrame,
        suite_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            suite_name: Name of expectation suite
            output_path: Output path for report
            
        Returns:
            Path to generated report
        """
        try:
            # Run validation
            validation_results = self.validate_data(df, suite_name)
            
            # Generate report
            if output_path is None:
                output_path = self.expectations_dir / f"data_quality_report_{suite_name}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': df.shape,
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'validation_results': validation_results,
                'data_summary': {
                    'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                    'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Data quality report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate data quality report: {e}")
            raise
    
    def list_expectation_suites(self) -> List[str]:
        """List all available expectation suites."""
        try:
            suites = self.context.list_expectation_suites()
            return [suite.expectation_suite_name for suite in suites]
        except Exception as e:
            logger.error(f"Failed to list expectation suites: {e}")
            return []
    
    def get_expectation_suite(self, suite_name: str) -> ExpectationSuite:
        """Get existing expectation suite."""
        try:
            return self.context.get_expectation_suite(suite_name)
        except Exception as e:
            logger.error(f"Failed to get expectation suite {suite_name}: {e}")
            raise


# Convenience functions
def validate_openfda_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate OpenFDA events data."""
    validator = DataValidator()
    return validator.validate_training_data(df, "serious", "openfda_events")


def validate_synthea_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Synthea events data."""
    validator = DataValidator()
    return validator.validate_training_data(df, "condition_code", "synthea_events")

