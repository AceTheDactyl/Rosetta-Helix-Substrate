"""Basic tests for data_validator."""

import pytest

from data_validator.core import DataValidator, create_data_validator
from data_validator.exceptions import DataValidatorError, ValidationError


class TestDataValidator:
    """Test suite for DataValidator."""
    
    def test_init(self):
        """Test basic initialization."""
        instance = DataValidator()
        assert instance is not None
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"key": "value"}
        instance = DataValidator(config=config)
        assert instance.config == config
    
    def test_factory(self):
        """Test factory function."""
        instance = create_data_validator(key="value")
        assert instance.config["key"] == "value"
    
    def test_validate(self):
        """Test validation."""
        instance = DataValidator()
        assert instance.validate() is True
    
    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        instance = DataValidator()
        with pytest.raises(NotImplementedError):
            instance.run()
