import re
import pytesseract
import cv2
import spacy
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor
from flair.models import SequenceTagger
from flair.data import Sentence

class PIIDetector:
    def __init__(self, tesseract_cmd_path: str):
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.flair_tagger = SequenceTagger.load("ner-fast")  # Faster NER model
        
        # Configure Tesseract path
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    # -------------------- Core OCR Methods --------------------
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        return pytesseract.image_to_string(Image.open(image_path))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF by converting pages to images."""
        images = convert_from_path(pdf_path , poppler_path=r"C:\poppler-24.08.0\Library\bin")
        # images = convert_from_path(pdf_path, poppler_path=r"C:\poppler\bin")

        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(self._extract_text_from_image, images))
        return " ".join(texts)

    def _extract_text_from_image(self, img) -> str:
        """Helper method for parallel OCR processing."""
        return pytesseract.image_to_string(img)

    # -------------------- PII Detection Methods --------------------
    def detect_structured_pii(self, text: str) -> dict:
        """Detect PAN, Aadhaar, emails, phones, and addresses using regex."""
        return {
            "Aadhaar": re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", text),
            "PAN": re.findall(r"[A-Z]{5}\d{4}[A-Z]{1}", text),
            "Email": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text),
            "Phone": re.findall(r"\b(?:\+91|0)?[6-9]\d{9}\b", text),
            "Address": re.findall(r"\b\d{1,5}\s[\w\s]{5,}\b", text)
        }

    def detect_names(self, text: str) -> list:
        """Detect names using spaCy's NER model."""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    def detect_names_flair(self, text: str) -> list:
        """Detect names using Flair's more accurate NER model."""
        sentence = Sentence(text)
        self.flair_tagger.predict(sentence)
        return [entity.text for entity in sentence.get_spans("ner") if entity.tag == "PER"]

    # -------------------- Redaction Methods --------------------
    def redact_text(self, text: str) -> str:
        """Mask PAN and Aadhaar numbers in text."""
        redacted = re.sub(r"([A-Z]{5})(\d{4})([A-Z]{1})", r"\1****\3", text)  # PAN
        return re.sub(r"\d{4}\s\d{4}\s\d{4}", "**** **** ****", redacted)  # Aadhaar

    def redact_image(self, image_path: str, output_path: str = "redacted.jpg"):
        """Draw black boxes over detected PII in original image."""
        img = cv2.imread(image_path)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        for i, text in enumerate(data["text"]):
            structured_pii = self.detect_structured_pii(text)
            names = self.detect_names(text)
            
            if any(structured_pii.values()) or names:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        cv2.imwrite(output_path, img)

    # -------------------- Compliance Methods --------------------
    def check_compliance(self, pii_data: dict, consent_given: bool = False) -> list:
        """Check GDPR/DPDP compliance issues."""
        issues = []
        if pii_data.get("Aadhaar") and not consent_given:
            issues.append("Aadhaar requires explicit consent under DPDP 2023")
        return issues