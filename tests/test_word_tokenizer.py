"""
Unit tests for the WordTokenizer class.

Tests tokenization accuracy, space preservation, and reconstruction fidelity
for various input scenarios including edge cases.
"""

import pytest
from shortest_combined_string.word_tokenizer import WordTokenizer
from shortest_combined_string.models import WordToken


class TestWordTokenizer:
    """Test cases for WordTokenizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = WordTokenizer()
    
    def test_simple_sentence_tokenization(self):
        """Test tokenization of a simple sentence with single spaces."""
        input_str = "hello world test"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 3
        
        # First word
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 0
        assert tokens[0].trailing_spaces == 0
        assert tokens[0].original_index == 0
        
        # Second word
        assert tokens[1].word == "world"
        assert tokens[1].leading_spaces == 1
        assert tokens[1].trailing_spaces == 0
        assert tokens[1].original_index == 1
        
        # Third word
        assert tokens[2].word == "test"
        assert tokens[2].leading_spaces == 1
        assert tokens[2].trailing_spaces == 0
        assert tokens[2].original_index == 2
    
    def test_leading_spaces_tokenization(self):
        """Test tokenization with leading spaces."""
        input_str = "  hello world"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 2
        
        # First word with leading spaces
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 2
        assert tokens[0].trailing_spaces == 0
        assert tokens[0].original_index == 0
        
        # Second word
        assert tokens[1].word == "world"
        assert tokens[1].leading_spaces == 1
        assert tokens[1].trailing_spaces == 0
        assert tokens[1].original_index == 1
    
    def test_trailing_spaces_tokenization(self):
        """Test tokenization with trailing spaces."""
        input_str = "hello world  "
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 2
        
        # First word
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 0
        assert tokens[0].trailing_spaces == 0
        assert tokens[0].original_index == 0
        
        # Last word with trailing spaces
        assert tokens[1].word == "world"
        assert tokens[1].leading_spaces == 1
        assert tokens[1].trailing_spaces == 2
        assert tokens[1].original_index == 1
    
    def test_multiple_spaces_tokenization(self):
        """Test tokenization with multiple spaces between words."""
        input_str = "hello   world    test"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 3
        
        # First word
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 0
        assert tokens[0].trailing_spaces == 0
        
        # Second word with multiple leading spaces
        assert tokens[1].word == "world"
        assert tokens[1].leading_spaces == 3
        assert tokens[1].trailing_spaces == 0
        
        # Third word with multiple leading spaces
        assert tokens[2].word == "test"
        assert tokens[2].leading_spaces == 4
        assert tokens[2].trailing_spaces == 0
    
    def test_complex_spacing_tokenization(self):
        """Test tokenization with complex spacing patterns."""
        input_str = "  hello   world    test  "
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 3
        
        # First word with leading spaces
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 2
        assert tokens[0].trailing_spaces == 0
        
        # Second word with multiple spaces
        assert tokens[1].word == "world"
        assert tokens[1].leading_spaces == 3
        assert tokens[1].trailing_spaces == 0
        
        # Last word with trailing spaces
        assert tokens[2].word == "test"
        assert tokens[2].leading_spaces == 4
        assert tokens[2].trailing_spaces == 2
    
    def test_single_word_tokenization(self):
        """Test tokenization of a single word."""
        input_str = "hello"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 1
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 0
        assert tokens[0].trailing_spaces == 0
        assert tokens[0].original_index == 0
    
    def test_single_word_with_spaces_tokenization(self):
        """Test tokenization of a single word with spaces."""
        input_str = "  hello  "
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 1
        assert tokens[0].word == "hello"
        assert tokens[0].leading_spaces == 2
        assert tokens[0].trailing_spaces == 2
        assert tokens[0].original_index == 0
    
    def test_empty_string_tokenization(self):
        """Test tokenization of empty string."""
        input_str = ""
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 0
    
    def test_only_spaces_tokenization(self):
        """Test tokenization of string with only spaces."""
        input_str = "   "
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 0
    
    def test_special_characters_tokenization(self):
        """Test tokenization with special characters in words."""
        input_str = "hello-world test_case"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 2
        assert tokens[0].word == "hello-world"
        assert tokens[1].word == "test_case"
    
    def test_numbers_and_punctuation_tokenization(self):
        """Test tokenization with numbers and punctuation."""
        input_str = "test123 hello! world?"
        tokens = self.tokenizer.tokenize(input_str)
        
        assert len(tokens) == 3
        assert tokens[0].word == "test123"
        assert tokens[1].word == "hello!"
        assert tokens[2].word == "world?"
    
    def test_reconstruction_simple(self):
        """Test reconstruction of simple tokenized string."""
        input_str = "hello world test"
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_with_leading_spaces(self):
        """Test reconstruction with leading spaces."""
        input_str = "  hello world"
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_with_trailing_spaces(self):
        """Test reconstruction with trailing spaces."""
        input_str = "hello world  "
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_complex_spacing(self):
        """Test reconstruction with complex spacing."""
        input_str = "  hello   world    test  "
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_single_word(self):
        """Test reconstruction of single word."""
        input_str = "hello"
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_single_word_with_spaces(self):
        """Test reconstruction of single word with spaces."""
        input_str = "  hello  "
        tokens = self.tokenizer.tokenize(input_str)
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == input_str
    
    def test_reconstruction_empty_list(self):
        """Test reconstruction from empty token list."""
        tokens = []
        reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
        
        assert reconstructed == ""
    
    def test_round_trip_fidelity(self):
        """Test that tokenization and reconstruction preserve original string."""
        test_cases = [
            "hello world",
            "  hello world  ",
            "hello   world",
            "  hello   world    test  ",
            "single",
            "  single  ",
            "",
            "test123 hello! world?",
            "a b c d e",
            "   a   b   c   ",
        ]
        
        for input_str in test_cases:
            tokens = self.tokenizer.tokenize(input_str)
            reconstructed = self.tokenizer.reconstruct_from_tokens(tokens)
            assert reconstructed == input_str, f"Round-trip failed for: '{input_str}'"
    
    def test_tokenize_invalid_input_type(self):
        """Test tokenize with invalid input type."""
        with pytest.raises(TypeError, match="input_str must be a string"):
            self.tokenizer.tokenize(123)
        
        with pytest.raises(TypeError, match="input_str must be a string"):
            self.tokenizer.tokenize(None)
    
    def test_reconstruct_invalid_input_type(self):
        """Test reconstruct_from_tokens with invalid input type."""
        with pytest.raises(TypeError, match="tokens must be a list"):
            self.tokenizer.reconstruct_from_tokens("not a list")
        
        with pytest.raises(TypeError, match="tokens must be a list"):
            self.tokenizer.reconstruct_from_tokens(None)
    
    def test_reconstruct_invalid_token_type(self):
        """Test reconstruct_from_tokens with invalid token objects."""
        invalid_tokens = ["not", "word", "tokens"]
        
        with pytest.raises(TypeError, match="All tokens must be WordToken objects"):
            self.tokenizer.reconstruct_from_tokens(invalid_tokens)
    
    def test_word_token_total_length_property(self):
        """Test that WordToken total_length property works correctly."""
        input_str = "  hello   world  "
        tokens = self.tokenizer.tokenize(input_str)
        
        # First token: "hello" with 2 leading spaces
        assert tokens[0].total_length == 7  # 2 + 5 + 0
        
        # Second token: "world" with 3 leading and 2 trailing spaces
        assert tokens[1].total_length == 10  # 3 + 5 + 2
    
    def test_original_index_assignment(self):
        """Test that original_index is correctly assigned."""
        input_str = "first second third fourth"
        tokens = self.tokenizer.tokenize(input_str)
        
        for i, token in enumerate(tokens):
            assert token.original_index == i
    
    def test_primary_test_case_tokenization(self):
        """Test tokenization of the primary algorithm test case."""
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        
        tokens1 = self.tokenizer.tokenize(s1)
        tokens2 = self.tokenizer.tokenize(s2)
        
        # Verify s1 tokenization
        expected_words1 = ["this", "is", "a", "red", "vase"]
        assert len(tokens1) == len(expected_words1)
        for i, token in enumerate(tokens1):
            assert token.word == expected_words1[i]
            assert token.original_index == i
        
        # Verify s2 tokenization
        expected_words2 = ["his", "son", "freddy", "love", "vase"]
        assert len(tokens2) == len(expected_words2)
        for i, token in enumerate(tokens2):
            assert token.word == expected_words2[i]
            assert token.original_index == i
        
        # Verify reconstruction
        assert self.tokenizer.reconstruct_from_tokens(tokens1) == s1
        assert self.tokenizer.reconstruct_from_tokens(tokens2) == s2