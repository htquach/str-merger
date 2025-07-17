"""
Test suite for verifying word boundary preservation in the ShortestCombinedString algorithm.

This test suite focuses specifically on ensuring that spaces between words are preserved
in the output, addressing Issue-01 where word boundaries were incorrectly removed.
"""

import pytest
from shortest_combined_string.shortest_combined_string import ShortestCombinedString
from shortest_combined_string.subsequence_verifier import SubsequenceVerifier


class TestWordBoundaryPreservation:
    """Test suite for word boundary preservation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ShortestCombinedString()
        self.verifier = SubsequenceVerifier()
    
    def test_primary_case_word_boundaries(self):
        """Test that the primary case preserves word boundaries."""
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        
        result = self.algorithm.combine(s1, s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Check that spaces are preserved between words from the same input string
        s1_words = s1.split()
        s2_words = s2.split()
        
        # Check s1 word boundaries
        for i in range(len(s1_words) - 1):
            word1 = s1_words[i]
            word2 = s1_words[i + 1]
            
            # Find positions of these words in the output
            pos1 = result.combined_string.find(word1)
            pos2 = result.combined_string.find(word2, pos1 + len(word1))
            
            assert pos1 != -1, f"Word '{word1}' not found in output"
            assert pos2 != -1, f"Word '{word2}' not found in output"
            
            # Check if there's at least one space between the words
            between_text = result.combined_string[pos1 + len(word1):pos2]
            assert any(c.isspace() for c in between_text), \
                f"Word boundary violation: '{word1}' and '{word2}' are not separated by spaces in output"
        
        # Check s2 word boundaries
        for i in range(len(s2_words) - 1):
            word1 = s2_words[i]
            word2 = s2_words[i + 1]
            
            # Find positions of these words in the output
            pos1 = result.combined_string.find(word1)
            pos2 = result.combined_string.find(word2, pos1 + len(word1))
            
            assert pos1 != -1, f"Word '{word1}' not found in output"
            assert pos2 != -1, f"Word '{word2}' not found in output"
            
            # Check if there's at least one space between the words
            between_text = result.combined_string[pos1 + len(word1):pos2]
            assert any(c.isspace() for c in between_text), \
                f"Word boundary violation: '{word1}' and '{word2}' are not separated by spaces in output"
        
        # Print the actual output for reference
        print(f"Output with preserved word boundaries: '{result.combined_string}'")
    
    def test_simple_word_boundary_case(self):
        """Test a simple case where word boundaries must be preserved."""
        s1 = "hello world"
        s2 = "world test"
        
        result = self.algorithm.combine(s1, s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Expected output should be "hello world test" with spaces preserved
        assert "hello world test" == result.combined_string, \
            f"Expected 'hello world test' but got '{result.combined_string}'"
    
    def test_issue_01_specific_case(self):
        """Test the specific case mentioned in Issue-01."""
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        
        result = self.algorithm.combine(s1, s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Check that the output is not "this isasonfreddylovevase" (the incorrect version)
        assert result.combined_string != "this isasonfreddylovevase", \
            "Output should not combine words by removing spaces"
        
        # Check that specific word pairs maintain spaces between them
        assert " is " in result.combined_string, "Space between 'is' and adjacent words is missing"
        assert " a " in result.combined_string, "Space between 'a' and adjacent words is missing"
        assert " son " in result.combined_string, "Space between 'son' and adjacent words is missing"
        assert " freddy " in result.combined_string, "Space between 'freddy' and adjacent words is missing"
        assert " love " in result.combined_string, "Space between 'love' and adjacent words is missing"
        
        # Print the actual output for reference
        print(f"Issue-01 fixed output: '{result.combined_string}'")
    
    def test_subsequence_verifier_word_boundaries(self):
        """Test that the SubsequenceVerifier correctly identifies word boundary violations."""
        s1 = "hello world"
        s2 = "world test"
        
        # Valid output with preserved word boundaries
        valid_output = "hello world test"
        verification = self.verifier.verify(s1, s2, valid_output)
        assert verification.is_valid, "Valid output with preserved word boundaries should pass verification"
        
        # Invalid output with violated word boundaries
        invalid_output = "helloworldtest"
        verification = self.verifier.verify(s1, s2, invalid_output)
        assert verification.is_invalid, "Output with violated word boundaries should fail verification"
        assert any("Word boundary violation" in error for error in verification.validation_errors), \
            "Validation errors should include word boundary violation"