"""
Unit tests for core data models.

This module tests all data classes for proper validation, serialization,
and edge case handling.
"""

import pytest
from shortest_combined_string.models import (
    WordToken,
    PreprocessedInput,
    DPState,
    CombinedToken,
    OptimizationMetrics,
    AlgorithmResult,
    Operation,
    TokenType
)


class TestWordToken:
    """Test cases for WordToken data class."""
    
    def test_valid_word_token(self):
        """Test creating a valid WordToken."""
        token = WordToken(
            word="hello",
            leading_spaces=1,
            trailing_spaces=2,
            original_index=0
        )
        assert token.word == "hello"
        assert token.leading_spaces == 1
        assert token.trailing_spaces == 2
        assert token.original_index == 0
        assert token.total_length == 8  # 5 + 1 + 2
    
    def test_word_token_zero_spaces(self):
        """Test WordToken with zero spaces."""
        token = WordToken(
            word="test",
            leading_spaces=0,
            trailing_spaces=0,
            original_index=5
        )
        assert token.total_length == 4
    
    def test_word_token_empty_word(self):
        """Test WordToken with empty word."""
        token = WordToken(
            word="",
            leading_spaces=1,
            trailing_spaces=1,
            original_index=0
        )
        assert token.total_length == 2
    
    def test_word_token_invalid_word_type(self):
        """Test WordToken with invalid word type."""
        with pytest.raises(TypeError, match="word must be a string"):
            WordToken(
                word=123,
                leading_spaces=0,
                trailing_spaces=0,
                original_index=0
            )
    
    def test_word_token_negative_leading_spaces(self):
        """Test WordToken with negative leading spaces."""
        with pytest.raises(ValueError, match="leading_spaces must be non-negative"):
            WordToken(
                word="test",
                leading_spaces=-1,
                trailing_spaces=0,
                original_index=0
            )
    
    def test_word_token_negative_trailing_spaces(self):
        """Test WordToken with negative trailing spaces."""
        with pytest.raises(ValueError, match="trailing_spaces must be non-negative"):
            WordToken(
                word="test",
                leading_spaces=0,
                trailing_spaces=-1,
                original_index=0
            )
    
    def test_word_token_negative_original_index(self):
        """Test WordToken with negative original index."""
        with pytest.raises(ValueError, match="original_index must be non-negative"):
            WordToken(
                word="test",
                leading_spaces=0,
                trailing_spaces=0,
                original_index=-1
            )


class TestPreprocessedInput:
    """Test cases for PreprocessedInput data class."""
    
    def test_valid_preprocessed_input(self):
        """Test creating a valid PreprocessedInput."""
        input_data = PreprocessedInput(
            s1="hello world",
            s2="goodbye world",
            warnings=["normalized consecutive spaces"],
            has_consecutive_spaces=True
        )
        assert input_data.s1 == "hello world"
        assert input_data.s2 == "goodbye world"
        assert len(input_data.warnings) == 1
        assert input_data.has_consecutive_spaces is True
    
    def test_preprocessed_input_no_warnings(self):
        """Test PreprocessedInput with no warnings."""
        input_data = PreprocessedInput(
            s1="test",
            s2="data",
            warnings=[],
            has_consecutive_spaces=False
        )
        assert len(input_data.warnings) == 0
        assert input_data.has_consecutive_spaces is False
    
    def test_preprocessed_input_invalid_s1_type(self):
        """Test PreprocessedInput with invalid s1 type."""
        with pytest.raises(TypeError, match="s1 must be a string"):
            PreprocessedInput(
                s1=123,
                s2="test",
                warnings=[],
                has_consecutive_spaces=False
            )
    
    def test_preprocessed_input_invalid_s2_type(self):
        """Test PreprocessedInput with invalid s2 type."""
        with pytest.raises(TypeError, match="s2 must be a string"):
            PreprocessedInput(
                s1="test",
                s2=123,
                warnings=[],
                has_consecutive_spaces=False
            )
    
    def test_preprocessed_input_invalid_warnings_type(self):
        """Test PreprocessedInput with invalid warnings type."""
        with pytest.raises(TypeError, match="warnings must be a list"):
            PreprocessedInput(
                s1="test",
                s2="data",
                warnings="not a list",
                has_consecutive_spaces=False
            )
    
    def test_preprocessed_input_invalid_has_consecutive_spaces_type(self):
        """Test PreprocessedInput with invalid has_consecutive_spaces type."""
        with pytest.raises(TypeError, match="has_consecutive_spaces must be a boolean"):
            PreprocessedInput(
                s1="test",
                s2="data",
                warnings=[],
                has_consecutive_spaces="not a boolean"
            )


class TestDPState:
    """Test cases for DPState data class."""
    
    def test_valid_dp_state(self):
        """Test creating a valid DPState."""
        state = DPState(
            length=10,
            s1_word_index=2,
            s2_word_index=3,
            operation=Operation.MATCH
        )
        assert state.length == 10
        assert state.s1_word_index == 2
        assert state.s2_word_index == 3
        assert state.operation == Operation.MATCH
    
    def test_dp_state_all_operations(self):
        """Test DPState with all operation types."""
        operations = [Operation.MATCH, Operation.INSERT_S1, Operation.INSERT_S2, Operation.SKIP]
        for op in operations:
            state = DPState(
                length=5,
                s1_word_index=1,
                s2_word_index=1,
                operation=op
            )
            assert state.operation == op
    
    def test_dp_state_negative_length(self):
        """Test DPState with negative length."""
        with pytest.raises(ValueError, match="length must be non-negative"):
            DPState(
                length=-1,
                s1_word_index=0,
                s2_word_index=0,
                operation=Operation.MATCH
            )
    
    def test_dp_state_negative_s1_word_index(self):
        """Test DPState with negative s1_word_index."""
        with pytest.raises(ValueError, match="s1_word_index must be non-negative"):
            DPState(
                length=0,
                s1_word_index=-1,
                s2_word_index=0,
                operation=Operation.MATCH
            )
    
    def test_dp_state_negative_s2_word_index(self):
        """Test DPState with negative s2_word_index."""
        with pytest.raises(ValueError, match="s2_word_index must be non-negative"):
            DPState(
                length=0,
                s1_word_index=0,
                s2_word_index=-1,
                operation=Operation.MATCH
            )
    
    def test_dp_state_invalid_operation_type(self):
        """Test DPState with invalid operation type."""
        with pytest.raises(TypeError, match="operation must be an Operation enum"):
            DPState(
                length=0,
                s1_word_index=0,
                s2_word_index=0,
                operation="INVALID"
            )


class TestCombinedToken:
    """Test cases for CombinedToken data class."""
    
    def test_valid_combined_token(self):
        """Test creating a valid CombinedToken."""
        token = CombinedToken(
            content="hello",
            source_s1_words=[0, 1],
            source_s2_words=[2],
            type=TokenType.MERGED
        )
        assert token.content == "hello"
        assert token.source_s1_words == [0, 1]
        assert token.source_s2_words == [2]
        assert token.type == TokenType.MERGED
    
    def test_combined_token_all_types(self):
        """Test CombinedToken with all token types."""
        types = [TokenType.MERGED, TokenType.S1_ONLY, TokenType.S2_ONLY, TokenType.SPACING]
        for token_type in types:
            token = CombinedToken(
                content="test",
                source_s1_words=[],
                source_s2_words=[],
                type=token_type
            )
            assert token.type == token_type
    
    def test_combined_token_empty_sources(self):
        """Test CombinedToken with empty source lists."""
        token = CombinedToken(
            content=" ",
            source_s1_words=[],
            source_s2_words=[],
            type=TokenType.SPACING
        )
        assert len(token.source_s1_words) == 0
        assert len(token.source_s2_words) == 0
    
    def test_combined_token_invalid_content_type(self):
        """Test CombinedToken with invalid content type."""
        with pytest.raises(TypeError, match="content must be a string"):
            CombinedToken(
                content=123,
                source_s1_words=[],
                source_s2_words=[],
                type=TokenType.MERGED
            )
    
    def test_combined_token_invalid_source_s1_words_type(self):
        """Test CombinedToken with invalid source_s1_words type."""
        with pytest.raises(TypeError, match="source_s1_words must be a list"):
            CombinedToken(
                content="test",
                source_s1_words="not a list",
                source_s2_words=[],
                type=TokenType.MERGED
            )
    
    def test_combined_token_invalid_source_s2_words_type(self):
        """Test CombinedToken with invalid source_s2_words type."""
        with pytest.raises(TypeError, match="source_s2_words must be a list"):
            CombinedToken(
                content="test",
                source_s1_words=[],
                source_s2_words="not a list",
                type=TokenType.MERGED
            )
    
    def test_combined_token_invalid_type_type(self):
        """Test CombinedToken with invalid type."""
        with pytest.raises(TypeError, match="type must be a TokenType enum"):
            CombinedToken(
                content="test",
                source_s1_words=[],
                source_s2_words=[],
                type="INVALID"
            )
    
    def test_combined_token_negative_source_indices(self):
        """Test CombinedToken with negative source indices."""
        with pytest.raises(ValueError, match="All source_s1_words indices must be non-negative integers"):
            CombinedToken(
                content="test",
                source_s1_words=[-1],
                source_s2_words=[],
                type=TokenType.S1_ONLY
            )
        
        with pytest.raises(ValueError, match="All source_s2_words indices must be non-negative integers"):
            CombinedToken(
                content="test",
                source_s1_words=[],
                source_s2_words=[-1],
                type=TokenType.S2_ONLY
            )


class TestOptimizationMetrics:
    """Test cases for OptimizationMetrics data class."""
    
    def test_valid_optimization_metrics(self):
        """Test creating valid OptimizationMetrics."""
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=15,
            combined_length=20,
            total_savings=5,
            compression_ratio=0.8
        )
        assert metrics.original_s1_length == 10
        assert metrics.original_s2_length == 15
        assert metrics.combined_length == 20
        assert metrics.total_savings == 5
        assert metrics.compression_ratio == 0.8
    
    def test_optimization_metrics_no_savings(self):
        """Test OptimizationMetrics with no savings."""
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=10,
            combined_length=20,
            total_savings=0,
            compression_ratio=1.0
        )
        assert metrics.total_savings == 0
        assert metrics.compression_ratio == 1.0
    
    def test_optimization_metrics_zero_lengths(self):
        """Test OptimizationMetrics with zero lengths."""
        metrics = OptimizationMetrics(
            original_s1_length=0,
            original_s2_length=0,
            combined_length=0,
            total_savings=0,
            compression_ratio=0.0
        )
        assert metrics.total_savings == 0
    
    def test_optimization_metrics_negative_original_s1_length(self):
        """Test OptimizationMetrics with negative original_s1_length."""
        with pytest.raises(ValueError, match="original_s1_length must be non-negative"):
            OptimizationMetrics(
                original_s1_length=-1,
                original_s2_length=10,
                combined_length=10,
                total_savings=0,
                compression_ratio=1.0
            )
    
    def test_optimization_metrics_negative_original_s2_length(self):
        """Test OptimizationMetrics with negative original_s2_length."""
        with pytest.raises(ValueError, match="original_s2_length must be non-negative"):
            OptimizationMetrics(
                original_s1_length=10,
                original_s2_length=-1,
                combined_length=10,
                total_savings=0,
                compression_ratio=1.0
            )
    
    def test_optimization_metrics_negative_combined_length(self):
        """Test OptimizationMetrics with negative combined_length."""
        with pytest.raises(ValueError, match="combined_length must be non-negative"):
            OptimizationMetrics(
                original_s1_length=10,
                original_s2_length=10,
                combined_length=-1,
                total_savings=0,
                compression_ratio=1.0
            )
    
    def test_optimization_metrics_negative_compression_ratio(self):
        """Test OptimizationMetrics with negative compression_ratio."""
        with pytest.raises(ValueError, match="compression_ratio must be non-negative"):
            OptimizationMetrics(
                original_s1_length=10,
                original_s2_length=10,
                combined_length=20,
                total_savings=0,
                compression_ratio=-0.5
            )
    
    def test_optimization_metrics_inconsistent_savings(self):
        """Test OptimizationMetrics with inconsistent total_savings."""
        with pytest.raises(ValueError, match="total_savings .* doesn't match calculated savings"):
            OptimizationMetrics(
                original_s1_length=10,
                original_s2_length=10,
                combined_length=15,
                total_savings=10,  # Should be 5
                compression_ratio=0.75
            )
    
    def test_optimization_metrics_inconsistent_compression_ratio(self):
        """Test OptimizationMetrics with inconsistent compression_ratio."""
        with pytest.raises(ValueError, match="compression_ratio .* doesn't match calculated ratio"):
            OptimizationMetrics(
                original_s1_length=10,
                original_s2_length=10,
                combined_length=15,
                total_savings=5,
                compression_ratio=0.5  # Should be 0.75
            )


class TestAlgorithmResult:
    """Test cases for AlgorithmResult data class."""
    

        
    def test_valid_algorithm_result_corrected(self):
        """Test creating a valid AlgorithmResult with correct lengths."""
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=10,
            combined_length=15,
            total_savings=5,
            compression_ratio=0.75
        )
        result = AlgorithmResult(
            combined_string="hello world tes",  # Exactly 15 chars
            metrics=metrics,
            is_valid=True,
            validation_errors=[],
            processing_warnings=["normalized spaces"]
        )
        assert result.combined_string == "hello world tes"
        assert result.is_valid is True
        assert len(result.validation_errors) == 0
        assert len(result.processing_warnings) == 1
    
    def test_algorithm_result_with_errors(self):
        """Test AlgorithmResult with validation errors."""
        metrics = OptimizationMetrics(
            original_s1_length=5,
            original_s2_length=5,
            combined_length=8,
            total_savings=2,
            compression_ratio=0.8
        )
        result = AlgorithmResult(
            combined_string="testdata",  # 8 chars
            metrics=metrics,
            is_valid=False,
            validation_errors=["subsequence validation failed"],
            processing_warnings=[]
        )
        assert result.is_valid is False
        assert len(result.validation_errors) == 1
    
    def test_algorithm_result_invalid_combined_string_type(self):
        """Test AlgorithmResult with invalid combined_string type."""
        metrics = OptimizationMetrics(
            original_s1_length=5,
            original_s2_length=5,
            combined_length=8,
            total_savings=2,
            compression_ratio=0.8
        )
        with pytest.raises(TypeError, match="combined_string must be a string"):
            AlgorithmResult(
                combined_string=123,
                metrics=metrics,
                is_valid=True,
                validation_errors=[],
                processing_warnings=[]
            )
    
    def test_algorithm_result_invalid_metrics_type(self):
        """Test AlgorithmResult with invalid metrics type."""
        with pytest.raises(TypeError, match="metrics must be an OptimizationMetrics instance"):
            AlgorithmResult(
                combined_string="test",
                metrics="not metrics",
                is_valid=True,
                validation_errors=[],
                processing_warnings=[]
            )
    
    def test_algorithm_result_invalid_is_valid_type(self):
        """Test AlgorithmResult with invalid is_valid type."""
        metrics = OptimizationMetrics(
            original_s1_length=4,
            original_s2_length=0,
            combined_length=4,
            total_savings=0,
            compression_ratio=1.0
        )
        with pytest.raises(TypeError, match="is_valid must be a boolean"):
            AlgorithmResult(
                combined_string="test",
                metrics=metrics,
                is_valid="not boolean",
                validation_errors=[],
                processing_warnings=[]
            )
    
    def test_algorithm_result_invalid_validation_errors_type(self):
        """Test AlgorithmResult with invalid validation_errors type."""
        metrics = OptimizationMetrics(
            original_s1_length=4,
            original_s2_length=0,
            combined_length=4,
            total_savings=0,
            compression_ratio=1.0
        )
        with pytest.raises(TypeError, match="validation_errors must be a list"):
            AlgorithmResult(
                combined_string="test",
                metrics=metrics,
                is_valid=True,
                validation_errors="not a list",
                processing_warnings=[]
            )
    
    def test_algorithm_result_invalid_processing_warnings_type(self):
        """Test AlgorithmResult with invalid processing_warnings type."""
        metrics = OptimizationMetrics(
            original_s1_length=4,
            original_s2_length=0,
            combined_length=4,
            total_savings=0,
            compression_ratio=1.0
        )
        with pytest.raises(TypeError, match="processing_warnings must be a list"):
            AlgorithmResult(
                combined_string="test",
                metrics=metrics,
                is_valid=True,
                validation_errors=[],
                processing_warnings="not a list"
            )
    
    def test_algorithm_result_length_mismatch(self):
        """Test AlgorithmResult with length mismatch between string and metrics."""
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=10,
            combined_length=15,  # Metrics say 15
            total_savings=5,
            compression_ratio=0.75
        )
        with pytest.raises(ValueError, match="combined_string length .* doesn't match metrics.combined_length"):
            AlgorithmResult(
                combined_string="test",  # But string is only 4 chars
                metrics=metrics,
                is_valid=True,
                validation_errors=[],
                processing_warnings=[]
            )