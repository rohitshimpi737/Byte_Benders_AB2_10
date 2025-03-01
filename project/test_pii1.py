from pii_detector1 import PIIDetector

def test_pii_detector():
    # Configure Tesseract path (Windows example)
    # tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_path = r"C:\Users\giris\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    
    # Initialize detector
    detector = PIIDetector(tesseract_path)
    
    # -------------------- Test Image Processing --------------------
    print("\n[1] Testing Image Processing...")
    # image_text = detector.extract_text("image.png")
    image_text = detector.extract_text("test.jpg")
    print("Extracted Text:\n", image_text[:200] + "...")  # Show first 200 chars
    
    # -------------------- Test PDF Processing --------------------
    # print("\n[2] Testing PDF Processing...")
    # pdf_text = detector.extract_text_from_pdf("resume.pdf")
    # print("PDF Text:\n", pdf_text[:200] + "...")
    
    # -------------------- Test PII Detection --------------------
    print("\n[3] Testing PII Detection...")
    pii_data = detector.detect_structured_pii(image_text)
    names = detector.detect_names(image_text)
    print("Structured PII:", {k: v for k, v in pii_data.items() if v})
    print("Names Detected:", names)
    
    # -------------------- Test Redaction --------------------
    print("\n[4] Testing Redaction...")
    detector.redact_image("test.jpg", "redacted.png")
    print("Image redaction completed → redacted.png")
    
    # -------------------- Test Compliance --------------------
    print("\n[5] Testing Compliance...")
    compliance_issues = detector.check_compliance(pii_data)
    print("Compliance Issues:", compliance_issues)

if __name__ == "__main__":
    test_pii_detector()