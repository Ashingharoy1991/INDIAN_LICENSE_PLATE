import re
from typing import Dict, List, Tuple

class SmartPlateTextCleaner:
    def __init__(self):
        # Indian license plate patterns
        self.indian_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # WB08A5504, WB08AB5504
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3}$',   # Shorter format
        ]
        
        # Common OCR confusions based on position and context
        self.position_based_corrections = {
            'letter_positions': [0, 1, 4, 5],  # Positions where letters are expected
            'digit_positions': [2, 3, 6, 7, 8, 9]  # Positions where digits are expected
        }
        
        # Character similarity mappings
        # self.letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'G': '5', 'B': '8'}
        self.letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'G': '5', 'B': '8','Z':'2'}
        # self.digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B'}
        self.digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
        
        
        # Valid Indian state codes (partial list)
        self.valid_state_codes = {
            'WB', 'MH', 'DL', 'KA', 'TN', 'AP', 'UP', 'MP', 'RJ', 'GJ', 
            'HR', 'PB', 'JH', 'BR', 'AS', 'OR', 'JK', 'HP', 'UK', 'CG'
        }

    def is_valid_indian_plate_format(self, text: str) -> bool:
        """Check if text matches Indian license plate format"""
        for pattern in self.indian_patterns:
            if re.match(pattern, text):
                return True
        return False

    def get_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on format compliance"""
        score = 0.0
        
        # Check length (typical Indian plates are 9-10 characters)
        if 9 <= len(text) <= 10:
            score += 0.3
        
        # Check if first two characters are valid state code
        if len(text) >= 2 and text[:2] in self.valid_state_codes:
            score += 0.4
        
        # Check if positions 2-3 are digits
        if len(text) >= 4 and text[2:4].isdigit():
            score += 0.2
        
        # Check if last 4 characters are digits
        if len(text) >= 4 and text[-4:].isdigit():
            score += 0.1
        
        return score

    def contextual_clean(self, text: str) -> str:
        """Apply contextual corrections based on expected format"""
        text = text.upper().strip()
        
        if len(text) < 9:
            return text
        
        cleaned = list(text)
        
        # Fix state code (first 2 positions - should be letters)
        for i in range(min(2, len(cleaned))):
            if cleaned[i].isdigit() and cleaned[i] in self.digit_to_letter:
                cleaned[i] = self.digit_to_letter[cleaned[i]]
        
        # Fix district code (positions 2-3 - should be digits)
        for i in range(2, min(4, len(cleaned))):
            if cleaned[i].isalpha() and cleaned[i] in self.letter_to_digit:
                cleaned[i] = self.letter_to_digit[cleaned[i]]
        
        # Fix series letters (position 4 and possibly 5)
        # Be more careful here - only correct obvious mistakes
        for i in range(4, min(6, len(cleaned))):
            if i < len(cleaned):
                # Only convert digits to letters if it's clearly wrong
                if cleaned[i].isdigit() and cleaned[i] in ['0', '1', '5', '6', '8']:
                    # Check if converting would make sense
                    potential_letter = self.digit_to_letter.get(cleaned[i])
                    if potential_letter and self._should_convert_to_letter(cleaned, i):
                        cleaned[i] = potential_letter
        
        # Fix number part (last 4 positions - should be digits)
        start_pos = max(0, len(cleaned) - 4)
        for i in range(start_pos, len(cleaned)):
            if i < len(cleaned) and cleaned[i].isalpha() and cleaned[i] in self.letter_to_digit:
                cleaned[i] = self.letter_to_digit[cleaned[i]]
        
        return ''.join(cleaned)

    def _should_convert_to_letter(self, text_list: List[str], position: int) -> bool:
        """Decide whether to convert a digit to letter at given position"""
        # If it's in the typical letter section (position 4-5) and 
        # the rest of the format looks correct, convert
        if position == 4:
            return True  # Position 4 should almost always be a letter
        elif position == 5:
            # Position 5 might be a letter (for newer format) or digit (for older format)
            # Look at the total length to decide
            if len(text_list) == 10:  # Newer format likely has 2 letters
                return True
            else:
                return False
        return False

    def multiple_candidate_approach(self, text: str) -> List[Tuple[str, float]]:
        """Generate multiple correction candidates and score them"""
        candidates = []
        
        # Original text
        original_score = self.get_confidence_score(text)
        candidates.append((text, original_score))
        
        # Contextually cleaned version
        cleaned = self.contextual_clean(text)
        cleaned_score = self.get_confidence_score(cleaned)
        candidates.append((cleaned, cleaned_score))
        
        # Additional variations for ambiguous characters
        variations = self._generate_variations(text)
        for variation in variations:
            score = self.get_confidence_score(variation)
            candidates.append((variation, score))
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates

    def _generate_variations(self, text: str) -> List[str]:
        """Generate variations by trying different interpretations of ambiguous characters"""
        variations = []
        text = text.upper().strip()
        
        # Focus on common confusion points
        # ambiguous_chars = {'0': 'O', 'O': '0', '5': 'S', 'S': '5', '6': 'G', 'G': '5'}
        ambiguous_chars = {'0': 'O', 'O': '0', '5': 'S', 'S': '5', 'G': '5'}
        
        for i, char in enumerate(text):
            if char in ambiguous_chars:
                variation = list(text)
                variation[i] = ambiguous_chars[char]
                variations.append(''.join(variation))
        
        return variations

    def smart_clean_plate_text(self, text: str, confidence_threshold: float = 0.5) -> str:
        """Main function to clean plate text intelligently"""
        candidates = self.multiple_candidate_approach(text)
        
        # Return the highest scoring candidate that meets threshold
        for candidate, score in candidates:
            if score >= confidence_threshold:
                return candidate
        
        # If no candidate meets threshold, return the best one
        return candidates[0][0] if candidates else text

    def clean_with_validation(self, text: str) -> Dict[str, any]:
        """Clean text and provide validation information"""
        candidates = self.multiple_candidate_approach(text)
        best_candidate = candidates[0][0] if candidates else text
        
        return {
            'original': text,
            'cleaned': best_candidate,
            'confidence': candidates[0][1] if candidates else 0.0,
            'is_valid_format': self.is_valid_indian_plate_format(best_candidate),
            'all_candidates': candidates[:3]  # Top 3 candidates
        }


# Integration with your existing code
def clean_plate_text_improved(text: str) -> str:
    """Improved version of your original function"""
    cleaner = SmartPlateTextCleaner()
    return cleaner.smart_clean_plate_text(text)

# Alternative: Get detailed validation info
def clean_plate_text_with_info(text: str) -> dict:
    """Clean text and get detailed validation information"""
    cleaner = SmartPlateTextCleaner()
    return cleaner.clean_with_validation(text)

# Replace your original clean_plate_text function with this:
def clean_plate_text(text):
    """Clean and format recognized license plate text - IMPROVED VERSION"""
    cleaner = SmartPlateTextCleaner()
    result = cleaner.clean_with_validation(text)
    
    # You can add additional logging here if needed
    if result['confidence'] < 0.5:
        print(f"Warning: Low confidence cleaning for '{text}' -> '{result['cleaned']}'")
    
    return result['cleaned']

# Test the improved function
if __name__ == "__main__":
    cleaner = SmartPlateTextCleaner()
    
    test_cases = [
        "WB08AG504",  # Should become WB08A6504 (G->6 in number position)
        "WB25GOO89",  # Should become WB25G0089 (OO->00 in number position)
        "WB25G0089",  # Should remain WB25G0089 (already correct)
        "WB08A5504",  # Should remain WB08A5504 (already correct)
        "DL1CA1234",  # Should remain DL1CA1234
        "MH12AB1234", # Should remain MH12AB1234
    ]
    
    for test in test_cases:
        result = cleaner.smart_clean_plate_text(test)
        print(f"Original: {test}")
        print(f"Cleaned: {result['cleaned']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Valid format: {result['is_valid_format']}")
        print(f"Top candidates: {result['all_candidates']}")
        print("-" * 50)