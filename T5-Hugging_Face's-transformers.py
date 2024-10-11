from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text):
    # Encode the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example legal text
legal_text = """
Legal Document: Agreement for Services

THIS AGREEMENT is made as of October 1, 2024, by and between John Doe Enterprises, having an address at 123 Business Rd, Suite 400, Springfield, IL 62701 (hereinafter referred to as "Client"), and XYZ Consulting LLC, having an address at 456 Consultant Ave, Suite 200, Chicago, IL 60601 (hereinafter referred to as "Service Provider").

1. SERVICES
The Service Provider agrees to provide the following services to the Client:

Consulting and strategic planning services related to business development.
Project management and implementation of operational processes.
2. TERM
This Agreement shall commence on October 15, 2024, and continue until October 15, 2025, unless terminated earlier in accordance with the provisions of this Agreement.

3. COMPENSATION
The Client agrees to pay the Service Provider the total amount of $50,000 for the services rendered under this Agreement. Payments shall be made according to the following schedule:

$25,000 due upon signing of this Agreement.
$25,000 due upon completion of services on October 15, 2025.
4. TERMINATION
Either party may terminate this Agreement upon written notice of 30 days to the other party. Upon termination, the Client shall pay for all services performed up to the termination date.

5. CONFIDENTIALITY
Both parties agree to maintain the confidentiality of all proprietary information disclosed during the term of this Agreement. This obligation shall survive the termination of this Agreement.

6. INDEMNIFICATION
The Service Provider agrees to indemnify and hold harmless the Client from any claims, damages, or liabilities arising from the performance of the services under this Agreement.

7. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of Illinois.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.

CLIENT
Signature: ________________________
Name: John Doe
Title: CEO
Date: October 1, 2024

SERVICE PROVIDER
Signature: ________________________
Name: Jane Smith
Title: Managing Partner
Date: October 1, 2024


"""

summary = summarize_text(legal_text)
print("Summary:", summary)