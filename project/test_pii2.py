from pii_detector2 import PIIDetector

def test_pii_detector():
    # Configure Tesseract & Poppler paths
    tesseract_path = r"C:\Users\giris\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    poppler_path = r"C:\poppler-24.08.0\Library\bin"
    
    # Initialize detector
    detector = PIIDetector(tesseract_path, poppler_path)
    
    # -------------------- Test Image Processing --------------------
    print("\n[1] Testing Image Processing...")
    image_text = detector.extract_text_from_image("image.png")
    print("Extracted Text:\n", image_text[:200] + "...")  # Show first 200 chars
    
    # -------------------- Test PDF Processing --------------------
    print("\n[2] Testing PDF Processing...")
    pdf_text = detector.extract_text_from_pdf("resume.pdf")
    print("PDF Text:\n", pdf_text[:200] + "...")
    
    # -------------------- Test PII Detection --------------------
    print("\n[3] Testing PII Detection...")
    pii_data = detector.detect_structured_pii(image_text)
    names = detector.detect_pii_nlp(image_text)
    print("Structured PII:", pii_data)
    print("Names Detected:", names)
    
    # -------------------- Test Redaction --------------------
    print("\n[4] Testing Redaction...")
    detector.redact_image("image.png", "redacted.png")
    print("Image redaction completed → redacted.png")
    
    # -------------------- Test Compliance --------------------
    print("\n[5] Testing Compliance...")
    compliance_issues = detector.check_compliance(pii_data)
    print("Compliance Issues:", compliance_issues)

if __name__ == "__main__":
    test_pii_detector()
