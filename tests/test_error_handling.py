"""
Unit tests for error handling and validation in the Shortest Combined String algorithm.

This module tests the error handling and validation functionality across all components
of the algorithm to ensure proper exception raising and error messages.
"""

import unittest
from shortest_combined_string.exceptions import (
    ShortestCombinedStringError, InputValidationError, TokenizationError,
    DPSolverError, PathReconstructionError, FormattingError, VerificationError
)
from shortest_combined_string.input_processor import InputProcessor
from shortest_combined_string.word_tokenizer import WordTokenizer
from shortest_combined_string.subsequence_verifier import SubsequenceVerifier
from shortest_combined_string.dp_solver import DPSolver
from shortest_combined_string.path_reconstructor import PathReconstructor
from shortest_combined_string.result_formatter import ResultFormatter
from shortest_combined_string.models import WordToken, DPState, Operation, CombinedToken, TokenType
from shortest_combined_string.shortest_combined_string import ShortestCombinedString


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling in InputProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_processor = InputProcessor()
    
    def test_none_inputs(self):
        """Test that None inputs raise InputValidationError."""
        with self.assertRaises(InputValidationError) as context:
            self.input_processor.validate_and_preprocess(None, "test")
        self.assertIn("cannot be None", str(context.exception))
        
        with self.assertRaises(InputValidationError) as context:
            self.input_processor.validate_and_preprocess("test", None)
        self.assertIn("cannot be None", str(context.exception))
    
    def test_non_string_inputs(self):
        """Test that non-string inputs raise InputValidationError."""
        with self.assertRaises(InputValidationError) as context:
            self.input_processor.validate_and_preprocess(123, "test")
        self.assertIn("must be a string", str(context.exception))
        
        with self.assertRaises(InputValidationError) as context:
            self.input_processor.validate_and_preprocess("test", [1, 2, 3])
        self.assertIn("must be a string", str(context.exception))
    
    def test_normalize_spaces_validation(self):
        """Test that normalize_spaces validates input types."""
        with self.assertRaises(InputValidationError) as context:
            self.input_processor.normalize_spaces(123)
        self.assertIn("must be a string", str(context.exception))


class TestTokenizationValidation(unittest.TestCase):
    """Test validation and error handling in WordTokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = WordTokenizer()
    
    def test_tokenize_non_string(self):
        """Test that tokenize validates input types."""
        with self.assertRaises(TokenizationError) as context:
            self.tokenizer.tokenize(123)
        self.assertIn("must be a string", str(context.exception))
    
    def test_reconstruct_non_list(self):
        """Test that reconstruct_from_tokens validates input types."""
        with self.assertRaises(TokenizationError) as context:
            self.tokenizer.reconstruct_from_tokens("not a list")
        self.assertIn("must be a list", str(context.exception))
    
    def test_reconstruct_invalid_tokens(self):
        """Test that reconstruct_from_tokens validates token types."""
        with self.assertRaises(TokenizationError) as context:
            self.tokenizer.reconstruct_from_tokens(["not a token", "also not a token"])
        self.assertIn("must be a WordToken object", str(context.exception))


class TestVerificationValidation(unittest.TestCase):
    """Test validation and error handling in SubsequenceVerifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = SubsequenceVerifier()
    
    def test_verify_non_string_inputs(self):
        """Test that verify validates input types."""
        with self.assertRaises(VerificationError) as context:
            self.verifier.verify(123, "test", "output")
        self.assertIn("must be a string", str(context.exception))
        
        with self.assertRaises(VerificationError) as context:
            self.verifier.verify("test", [1, 2, 3], "output")
        self.assertIn("must be a string", str(context.exception))
        
        with self.assertRaises(VerificationError) as context:
            self.verifier.verify("test", "test", {"key": "value"})
        self.assertIn("must be a string", str(context.exception))


class TestDPSolverValidation(unittest.TestCase):
    """Test validation and error handling in DPSolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = DPSolver()
        self.valid_token = WordToken(word="test", leading_spaces=0, trailing_spaces=0, original_index=0)
    
    def test_solve_non_list_inputs(self):
        """Test that solve validates input types."""
        with self.assertRaises(DPSolverError) as context:
            self.solver.solve("not a list", [self.valid_token])
        self.assertIn("must be a list", str(context.exception))
        
        with self.assertRaises(DPSolverError) as context:
            self.solver.solve([self.valid_token], "not a list")
        self.assertIn("must be a list", str(context.exception))
    
    def test_solve_invalid_tokens(self):
        """Test that solve validates token types."""
        with self.assertRaises(DPSolverError) as context:
            self.solver.solve(["not a token"], [self.valid_token])
        self.assertIn("must be a WordToken object", str(context.exception))
        
        with self.assertRaises(DPSolverError) as context:
            self.solver.solve([self.valid_token], [123])
        self.assertIn("must be a WordToken object", str(context.exception))


class TestPathReconstructorValidation(unittest.TestCase):
    """Test validation and error handling in PathReconstructor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reconstructor = PathReconstructor()
        self.valid_token = WordToken(word="test", leading_spaces=0, trailing_spaces=0, original_index=0)
        self.valid_state = DPState(length=0, s1_word_index=0, s2_word_index=0, operation=Operation.SKIP)
    
    def test_reconstruct_path_invalid_inputs(self):
        """Test that reconstruct_path validates input types."""
        # Create a valid DP table for testing
        valid_dp_table = [[self.valid_state]]
        
        with self.assertRaises(PathReconstructionError) as context:
            self.reconstructor.reconstruct_path("not a list", [self.valid_token], [self.valid_token])
        self.assertIn("must be a list", str(context.exception))
        
        with self.assertRaises(PathReconstructionError) as context:
            self.reconstructor.reconstruct_path(valid_dp_table, "not a list", [self.valid_token])
        self.assertIn("must be a list", str(context.exception))
        
        with self.assertRaises(PathReconstructionError) as context:
            self.reconstructor.reconstruct_path(valid_dp_table, [self.valid_token], "not a list")
        self.assertIn("must be a list", str(context.exception))
    
    def test_reconstruct_path_invalid_dp_table(self):
        """Test that reconstruct_path validates DP table structure."""
        # Test with invalid DP table dimensions
        with self.assertRaises(PathReconstructionError) as context:
            self.reconstructor.reconstruct_path([[self.valid_state]], [self.valid_token, self.valid_token], [self.valid_token])
        self.assertIn("DP table rows", str(context.exception))
        
        # Test with invalid DP table contents
        with self.assertRaises(PathReconstructionError) as context:
            self.reconstructor.reconstruct_path([["not a state"]], [], [])
        self.assertIn("must be a DPState object", str(context.exception))


class TestResultFormatterValidation(unittest.TestCase):
    """Test validation and error handling in ResultFormatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter()
        self.valid_token = CombinedToken(
            content="test",
            source_s1_words=[0],
            source_s2_words=[0],
            type=TokenType.MERGED
        )
    
    def test_format_result_invalid_inputs(self):
        """Test that format_result validates input types."""
        with self.assertRaises(FormattingError) as context:
            self.formatter.format_result("not a list", "s1", "s2")
        self.assertIn("must be a list", str(context.exception))
        
        with self.assertRaises(FormattingError) as context:
            self.formatter.format_result([self.valid_token], 123, "s2")
        self.assertIn("must be a string", str(context.exception))
        
        with self.assertRaises(FormattingError) as context:
            self.formatter.format_result([self.valid_token], "s1", [1, 2, 3])
        self.assertIn("must be a string", str(context.exception))
    
    def test_format_result_invalid_tokens(self):
        """Test that format_result validates token types."""
        with self.assertRaises(FormattingError) as context:
            self.formatter.format_result(["not a token"], "s1", "s2")
        self.assertIn("must be a CombinedToken object", str(context.exception))


class TestShortestCombinedStringValidation(unittest.TestCase):
    """Test validation and error handling in the main algorithm orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.algorithm = ShortestCombinedString()
    
    def test_combine_invalid_inputs(self):
        """Test that combine validates input types."""
        with self.assertRaises(InputValidationError) as context:
            self.algorithm.combine(None, "test")
        self.assertIn("cannot be None", str(context.exception))
        
        with self.assertRaises(InputValidationError) as context:
            self.algorithm.combine("test", None)
        self.assertIn("cannot be None", str(context.exception))
        
        with self.assertRaises(InputValidationError) as context:
            self.algorithm.combine(123, "test")
        self.assertIn("must be a string", str(context.exception))
        
        with self.assertRaises(InputValidationError) as context:
            self.algorithm.combine("test", [1, 2, 3])
        self.assertIn("must be a string", str(context.exception))
    
    def test_edge_case_handling(self):
        """Test that edge case handling works correctly."""
        # Test with empty strings
        result = self.algorithm.combine("", "")
        self.assertEqual(result.combined_string, "")
        self.assertTrue(result.is_valid)
        
        # Test with identical strings
        result = self.algorithm.combine("test", "test")
        self.assertEqual(result.combined_string, "test")
        self.assertTrue(result.is_valid)
        
        # Test with one string containing the other
        result = self.algorithm.combine("test", "te")
        self.assertEqual(result.combined_string, "test")
        self.assertTrue(result.is_valid)


if __name__ == '__main__':
    unittest.main()