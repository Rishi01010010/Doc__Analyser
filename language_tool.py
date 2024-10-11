import language_tool_python

def correct_text(input_text):
    # Initialize LanguageTool
    tool = language_tool_python.LanguageTool('en-US')
    
    # Check the text for errors
    matches = tool.check(input_text)
    
    # Correct the text
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    
    return corrected_text

# Input text
input_text = "This is smple text a speling error."

# Get corrected text
corrected_output = correct_text(input_text)

print("Original Text:", input_text)
print("Corrected Text:", corrected_output)