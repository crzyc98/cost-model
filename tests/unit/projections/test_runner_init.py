"""
Tests for the projection runner initialization module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from cost_model.projections.runner.init import initialize

class TestRunnerInit(unittest.TestCase):
    
    def setUp(self):
        self.config_ns = {
            "config": "test_config.yaml",
            "start_year": 2025,
            "num_years": 5
        }
        
        self.initial_snapshot = pd.DataFrame({
            "employee_id": ["1", "2"],
            "employee_hire_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "active": [True, True]
        })
        
        self.initial_log = pd.DataFrame({"event_id": ["1"]})
        
    @patch("cost_model.projections.runner.init.load_yaml_config")
    @patch("cost_model.projections.runner.init.parse_config")
    def test_initialize(self, mock_parse_config, mock_load_config):
        # Mock the config loading and parsing
        mock_config = {
            "seed": 42,
            "start_year": 2025,
            "num_years": 5
        }
        mock_load_config.return_value = mock_config
        mock_parse_config.return_value = ({"seed": 42}, {"rules": "test"})
        
        result = initialize(self.config_ns, self.initial_snapshot, self.initial_log)
        
        # Verify the results
        global_params, plan_rules, rng, years, census_path, ee_contrib_event_types = result
        
        self.assertEqual(global_params["seed"], 42)
        self.assertEqual(plan_rules["rules"], "test")
        self.assertIsInstance(rng, np.random.Generator)
        self.assertEqual(years, list(range(2025, 2030)))
        self.assertEqual(ee_contrib_event_types, ["EVT_CONTRIB"])
        
        # Verify config loading and parsing were called
        mock_load_config.assert_called_once_with("test_config.yaml")
        mock_parse_config.assert_called_once_with(mock_config)
        
    def test_invalid_snapshot(self):
        # Test with invalid snapshot (missing required columns)
        invalid_snapshot = pd.DataFrame({"employee_id": ["1"]})
        with self.assertRaises(ValueError):
            initialize(self.config_ns, invalid_snapshot, self.initial_log)
            
        # Test with invalid snapshot (non-unique EMP_IDs)
        invalid_snapshot = pd.DataFrame({
            "employee_id": ["1", "1"],
            "employee_hire_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")]
        })
        with self.assertRaises(ValueError):
            initialize(self.config_ns, invalid_snapshot, self.initial_log)
            
if __name__ == "__main__":
    unittest.main()
