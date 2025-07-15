"""
Unit tests for the InputProcessor class.

Tests cover edge cases including empty strings, only spaces, mixed content,
consecutive space detection, normalization, and error handling.
"""

import pytest
from shortest_combined_string.input_processor import InputProcessor
from shortest_combined_string.models import PreprocessedInput


class TestInputProcessor:
    """Test cases for InputProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = InputProcessor()
    
    def test_init(self):
        """Test InputProcessor initialization."""
        processor = InputProcessor()
        assert processor is not None
        assert hasattr(processor, '_consecutive_spaces_pattern')
    
    def test_validate_and_preprocess_normal_strings(self):
        """Test preprocessing with normal strings without consecutive spaces."""
        s1 = "hello world"
        s2 = "goodbye moon"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert isinstance(result, PreprocessedInput)
        assert result.s1 == s1
        assert result.s2 == s2
        assert result.warnings == []
        assert result.has_consecutive_spaces is False
    
    def test_validate_and_preprocess_empty_strings(self):
        """Test preprocessing with empty strings."""
        result = self.processor.validate_and_preprocess("", "")
        
        assert result.s1 == ""
        assert result.s2 == ""
        assert result.warnings == []
        assert result.has_consecutive_spaces is False
    
    def test_validate_and_preprocess_one_empty_string(self):
        """Test preprocessing with one empty string."""
        result = self.processor.validate_and_preprocess("hello", "")
        
        assert result.s1 == "hello"
        assert result.s2 == ""
        assert result.warnings == []
        assert result.has_consecutive_spaces is False
    
    def test_validate_and_preprocess_only_spaces(self):
        """Test preprocessing with strings containing only spaces."""
        result = self.processor.validate_and_preprocess("   ", "  ")
        
        assert result.s1 == " "  # Normalized to single space
        assert result.s2 == " "  # Normalized to single space
        assert len(result.warnings) == 2
        assert result.has_consecutive_spaces is True
        assert "s1" in result.warnings[0]
        assert "s2" in result.warnings[1]
    
    def test_validate_and_preprocess_consecutive_spaces_s1(self):
        """Test preprocessing with consecutive spaces in first string."""
        s1 = "hello  world"  # Two spaces
        s2 = "goodbye moon"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == "hello world"  # Normalized
        assert result.s2 == s2
        assert len(result.warnings) == 1
        assert result.has_consecutive_spaces is True
        assert "s1" in result.warnings[0]
        assert "position 5" in result.warnings[0]
        assert "2 spaces" in result.warnings[0]
    
    def test_validate_and_preprocess_consecutive_spaces_s2(self):
        """Test preprocessing with consecutive spaces in second string."""
        s1 = "hello world"
        s2 = "goodbye   moon"  # Three spaces
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == s1
        assert result.s2 == "goodbye moon"  # Normalized
        assert len(result.warnings) == 1
        assert result.has_consecutive_spaces is True
        assert "s2" in result.warnings[0]
        assert "position 7" in result.warnings[0]
        assert "3 spaces" in result.warnings[0]
    
    def test_validate_and_preprocess_consecutive_spaces_both(self):
        """Test preprocessing with consecutive spaces in both strings."""
        s1 = "hello  world"  # Two spaces
        s2 = "goodbye    moon"  # Four spaces
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == "hello world"
        assert result.s2 == "goodbye moon"
        assert len(result.warnings) == 2
        assert result.has_consecutive_spaces is True
        assert "s1" in result.warnings[0]
        assert "s2" in result.warnings[1]
    
    def test_validate_and_preprocess_multiple_consecutive_spaces(self):
        """Test preprocessing with multiple instances of consecutive spaces."""
        s1 = "hello  world  test"  # Two instances
        s2 = "goodbye moon"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == "hello world test"
        assert result.s2 == s2
        assert len(result.warnings) == 1
        assert result.has_consecutive_spaces is True
        assert "position 5" in result.warnings[0]
        assert "position 12" in result.warnings[0]
    
    def test_validate_and_preprocess_mixed_content(self):
        """Test preprocessing with mixed content including numbers and symbols."""
        s1 = "test123  @#$"
        s2 = "hello   world!"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == "test123 @#$"
        assert result.s2 == "hello world!"
        assert len(result.warnings) == 2
        assert result.has_consecutive_spaces is True
    
    def test_validate_and_preprocess_leading_trailing_spaces(self):
        """Test preprocessing with leading and trailing spaces."""
        s1 = " hello  world "
        s2 = " goodbye moon "  # Single spaces, no normalization needed
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == " hello world "  # Leading/trailing preserved, middle normalized
        assert result.s2 == " goodbye moon "  # Only single spaces, no normalization needed
        assert len(result.warnings) == 1  # Only s1 has consecutive spaces
        assert result.has_consecutive_spaces is True
    
    def test_validate_and_preprocess_none_inputs(self):
        """Test preprocessing with None inputs."""
        with pytest.raises(ValueError, match="Input strings cannot be None"):
            self.processor.validate_and_preprocess(None, "test")
        
        with pytest.raises(ValueError, match="Input strings cannot be None"):
            self.processor.validate_and_preprocess("test", None)
        
        with pytest.raises(ValueError, match="Input strings cannot be None"):
            self.processor.validate_and_preprocess(None, None)
    
    def test_validate_and_preprocess_non_string_inputs(self):
        """Test preprocessing with non-string inputs."""
        with pytest.raises(TypeError, match="Both inputs must be strings"):
            self.processor.validate_and_preprocess(123, "test")
        
        with pytest.raises(TypeError, match="Both inputs must be strings"):
            self.processor.validate_and_preprocess("test", 456)
        
        with pytest.raises(TypeError, match="Both inputs must be strings"):
            self.processor.validate_and_preprocess([], {})
    
    def test_has_consecutive_spaces_true_cases(self):
        """Test has_consecutive_spaces method with strings that have consecutive spaces."""
        test_cases = [
            "hello  world",
            "test   case",
            "   leading",
            "trailing   ",
            "  multiple  instances  ",
            "a    b",
        ]
        
        for test_case in test_cases:
            assert self.processor.has_consecutive_spaces(test_case) is True, f"Failed for: '{test_case}'"
    
    def test_has_consecutive_spaces_false_cases(self):
        """Test has_consecutive_spaces method with strings that don't have consecutive spaces."""
        test_cases = [
            "hello world",
            "test case",
            " single leading",
            "single trailing ",
            " single spaces only ",
            "nospaces",
            "",
            " ",
        ]
        
        for test_case in test_cases:
            assert self.processor.has_consecutive_spaces(test_case) is False, f"Failed for: '{test_case}'"
    
    def test_has_consecutive_spaces_non_string(self):
        """Test has_consecutive_spaces method with non-string input."""
        assert self.processor.has_consecutive_spaces(None) is False
        assert self.processor.has_consecutive_spaces(123) is False
        assert self.processor.has_consecutive_spaces([]) is False
    
    def test_normalize_spaces_basic(self):
        """Test normalize_spaces method with basic cases."""
        test_cases = [
            ("hello  world", "hello world"),
            ("test   case", "test case"),
            ("   leading", " leading"),
            ("trailing   ", "trailing "),
            ("  multiple  instances  ", " multiple instances "),
            ("no consecutive spaces", "no consecutive spaces"),
            ("", ""),
            (" ", " "),
        ]
        
        for input_str, expected in test_cases:
            result = self.processor.normalize_spaces(input_str)
            assert result == expected, f"Failed for '{input_str}': expected '{expected}', got '{result}'"
    
    def test_normalize_spaces_extreme_cases(self):
        """Test normalize_spaces method with extreme cases."""
        # Many consecutive spaces
        result = self.processor.normalize_spaces("a" + " " * 10 + "b")
        assert result == "a b"
        
        # Only consecutive spaces
        result = self.processor.normalize_spaces(" " * 5)
        assert result == " "
        
        # Mixed with other whitespace (should only affect spaces)
        result = self.processor.normalize_spaces("hello\t\tworld  test")
        assert result == "hello\t\tworld test"
    
    def test_normalize_spaces_non_string(self):
        """Test normalize_spaces method with non-string input."""
        with pytest.raises(TypeError, match="Input must be a string"):
            self.processor.normalize_spaces(None)
        
        with pytest.raises(TypeError, match="Input must be a string"):
            self.processor.normalize_spaces(123)
    
    def test_warning_message_format(self):
        """Test that warning messages contain expected information."""
        s1 = "hello  world   test"
        s2 = "normal string"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        warning = result.warnings[0]
        assert "s1" in warning
        assert "position 5" in warning
        assert "2 spaces" in warning
        assert "position 12" in warning  # Corrected position
        assert "3 spaces" in warning
        assert "Automatically normalized" in warning
    
    def test_edge_case_single_character_strings(self):
        """Test preprocessing with single character strings."""
        result = self.processor.validate_and_preprocess("a", "b")
        
        assert result.s1 == "a"
        assert result.s2 == "b"
        assert result.warnings == []
        assert result.has_consecutive_spaces is False
    
    def test_edge_case_unicode_strings(self):
        """Test preprocessing with unicode strings."""
        s1 = "héllo  wörld"
        s2 = "测试  字符串"
        
        result = self.processor.validate_and_preprocess(s1, s2)
        
        assert result.s1 == "héllo wörld"
        assert result.s2 == "测试 字符串"
        assert len(result.warnings) == 2
        assert result.has_consecutive_spaces is True