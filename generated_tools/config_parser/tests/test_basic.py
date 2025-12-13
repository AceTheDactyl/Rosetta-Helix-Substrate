"""Basic tests for config_parser."""

import pytest

from config_parser.core import ConfigParser, create_config_parser
from config_parser.exceptions import ConfigParserError, ValidationError


class TestConfigParser:
    """Test suite for ConfigParser."""
    
    def test_init(self):
        """Test basic initialization."""
        instance = ConfigParser()
        assert instance is not None
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"key": "value"}
        instance = ConfigParser(config=config)
        assert instance.config == config
    
    def test_factory(self):
        """Test factory function."""
        instance = create_config_parser(key="value")
        assert instance.config["key"] == "value"
    
    def test_validate(self):
        """Test validation."""
        instance = ConfigParser()
        assert instance.validate() is True
    
    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        instance = ConfigParser()
        with pytest.raises(NotImplementedError):
            instance.run()
