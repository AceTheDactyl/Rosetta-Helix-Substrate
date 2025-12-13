"""Basic tests for log_analyzer."""

import pytest

from log_analyzer.core import LogAnalyzer, create_log_analyzer
from log_analyzer.exceptions import LogAnalyzerError, ValidationError


class TestLogAnalyzer:
    """Test suite for LogAnalyzer."""
    
    def test_init(self):
        """Test basic initialization."""
        instance = LogAnalyzer()
        assert instance is not None
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"key": "value"}
        instance = LogAnalyzer(config=config)
        assert instance.config == config
    
    def test_factory(self):
        """Test factory function."""
        instance = create_log_analyzer(key="value")
        assert instance.config["key"] == "value"
    
    def test_validate(self):
        """Test validation."""
        instance = LogAnalyzer()
        assert instance.validate() is True
    
    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        instance = LogAnalyzer()
        with pytest.raises(NotImplementedError):
            instance.run()
