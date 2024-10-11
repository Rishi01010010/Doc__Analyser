import os
import sys
from PIL import Image
import pytesseract
import language_tool_python
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

def extract_text(image_path):
    """
    Extract text from an image using pytesseract.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text.
    """
    # Verify that the image exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file was not found: {image_path}")

    # Load the image using PIL
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise IOError(f"Error opening image file: {e}")

    # Specify the exact path to the tesseract executable if needed
    # Uncomment and update the path below if Tesseract is not in your PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Perform OCR using pytesseract
    try:
        text = pytesseract.image_to_string(image)
        print("\n--- Extracted Text ---\n")
        print(text)
        return text
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise EnvironmentError("Tesseract OCR executable not found. Please ensure it is installed and added to PATH.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during OCR: {e}")

def correct_text(input_text):
    """
    Correct grammatical and spelling errors in the input text using LanguageTool.

    Args:
        input_text (str): Text to be corrected.

    Returns:
        str: Corrected text.
    """
    # Initialize LanguageTool
    tool = language_tool_python.LanguageTool('en-US')

    # Check the text for errors
    matches = tool.check(input_text)

    # Correct the text
    corrected_text = language_tool_python.utils.correct(input_text, matches)

    print("\n--- Corrected Text ---\n")
    print(corrected_text)
    return corrected_text

def summarize_text(text):
    """
    Summarize the input text using a pre-trained PEGASUS model.

    Args:
        text (str): Text to be summarized.

    Returns:
        str: Summarized text.
    """
    # Load pre-trained PEGASUS model and tokenizer
    model_name = "google/pegasus-large"  # Using PEGASUS instead of T5
    try:
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading PEGASUS model: {e}")

    # Prepare the text for summarization
    preprocessed_text = text.strip().replace("\n", " ")

    # Tokenize and encode the input text
    inputs = tokenizer(preprocessed_text, truncation=True, padding="longest", return_tensors="pt")

    # Generate summary
    try:
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=150,    # Adjust as needed
            min_length=40,
            length_penalty=2.0,
            num_beams=5,
            early_stopping=True
        )
    except Exception as e:
        raise RuntimeError(f"Error during summarization: {e}")

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n--- Summarized Text ---\n")
    print(summary)
    return summary

def main():
    """
    Main function to process the image and extract, correct, and summarize text.
    """
    if len(sys.argv) != 2:
        print("Usage: python image_text_processor_pegasus.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Step 1: Extract text from image
        extracted_text = extract_text(image_path)

        if not extracted_text.strip():
            print("No text found in the image.")
            sys.exit(0)

        # Step 2: Correct the extracted text
        corrected_text = correct_text(extracted_text)

        # Step 3: Summarize the corrected text
        summarized_text = summarize_text(corrected_text)

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


# python -u "E:\My_Projects\Doc_Analyzer\Full-Google-Pegasus-integrated.py" "E:\My_Projects\Doc_Analyzer\agreement.jpg"