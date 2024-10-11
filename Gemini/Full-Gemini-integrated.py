import os
import sys
from PIL import Image
import pytesseract
import language_tool_python
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Ensure the API key is loaded correctly
if not gemini_api_key:
    raise EnvironmentError("Google Gemini API key not found in environment variables.")

# Configure the Google Gemini API
genai.configure(api_key=gemini_api_key)

# Setup the generation configuration for summarization
generation_config = {
    "temperature": 0.5,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

def extract_text(image_path):
    """
    Extract text from an image using pytesseract.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted text.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file was not found: {image_path}")

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print("\n--- Extracted Text ---\n")
        print(text)
        return text
    except Exception as e:
        raise IOError(f"Error extracting text from image: {e}")

def correct_text(input_text):
    """
    Correct grammatical and spelling errors in the input text using LanguageTool.
    Args:
        input_text (str): Text to be corrected.
    Returns:
        str: Corrected text.
    """
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    
    print("\n--- Corrected Text ---\n")
    print(corrected_text)
    return corrected_text

def summarize_text_with_gemini(text):
    """
    Summarize the input text using Google Gemini's Flash 5 Model.
    Args:
        text (str): Text to be summarized.
    Returns:
        str: Summarized text.
    """
    # Create the prompt for summarization
    summary_prompt = f"Refer and analyze the given text and draw a simple summary out of this text in simpler words:\n\n{text}"

    # Initialize the model and chat session
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    chat_session = model.start_chat(history=[])

    # Send the prompt to the model
    summary_response = chat_session.send_message(summary_prompt)
    summarized_text = summary_response.text

    print("\n--- Summarized Text ---\n")
    print(summarized_text)
    return summarized_text

def main():
    """
    Main function to process the image, correct, and summarize text.
    """
    if len(sys.argv) != 2:
        print("Usage: python gemini_image_text_processor.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Step 1: Extract text from the image
        extracted_text = extract_text(image_path)

        if not extracted_text.strip():
            print("No text found in the image.")
            sys.exit(0)

        # Step 2: Correct the extracted text
        corrected_text = correct_text(extracted_text)

        # Step 3: Summarize the corrected text using Google Gemini
        summarized_text = summarize_text_with_gemini(corrected_text)

        # Optionally, you can save the outputs to files
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(extracted_text)
        with open("corrected_text.txt", "w", encoding="utf-8") as f:
            f.write(corrected_text)
        with open("summarized_text.txt", "w", encoding="utf-8") as f:
            f.write(summarized_text)

        print("\nAll outputs have been saved to 'extracted_text.txt', 'corrected_text.txt', and 'summarized_text.txt'.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



# python -u "E:\My_Projects\Doc_Analyzer\Gemini\Full-Gemini-integrated.py" "E:\My_Projects\Doc_Analyzer\agreement.jpg"