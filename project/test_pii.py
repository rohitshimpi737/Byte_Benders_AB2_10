from pii_detector import PIIDetector

def test_pii_detector():
    # detector = PIIDetector()

    tesseract_cmd_path = r"C:\Users\giris\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    detector = PIIDetector(tesseract_cmd_path)
    
    # Step 1: OCR Extraction
    image_path = "image.png"  # Replace with your test image
    extracted_text = detector.extract_text(image_path)
    print("Extracted Text:\n", extracted_text)


    text = detector.extract_text_from_pdf("resume.pdf")
    print("\nExtracted Text from PDF:\n", text)
    
    # Step 2: Detect Structured PII (PAN, Aadhaar, Email)
    structured_pii = detector.detect_structured_pii(extracted_text)
    print("\nStructured PII:", structured_pii)
    
    # Step 3: Detect Names
    names = detector.detect_names(extracted_text)
    print("Names Detected:", names)
    
    # Step 4: Redact Sensitive Data
    redacted_text = detector.redact_text(extracted_text)
    print("\nRedacted Text:\n", redacted_text)

if __name__ == "__main__":
    test_pii_detector()