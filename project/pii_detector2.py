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
from transformers import pipeline
import os

class PIIDetector:
    def __init__(self, tesseract_cmd_path: str, poppler_path: str):
        # Initialize NLP models
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.flair_tagger = SequenceTagger.load("ner-fast")  # Fast Flair NER
        self.hf_pii_model = pipeline("ner", model="obi/deid_roberta_i2b2")  # Hugging Face PII model

        # Configure paths
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        self.poppler_path = poppler_path

    # -------------------- Image Preprocessing --------------------
    def preprocess_image(self, image_path: str):
        """Preprocess image to improve OCR accuracy (grayscale, thresholding, noise removal)."""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]  # Binarization
        noise_removed = cv2.medianBlur(thresh, 3)  # Remove small noise
        return noise_removed

    # -------------------- Core OCR Methods --------------------
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR ."""
        preprocessed_img = self.preprocess_image(image_path)
        return pytesseract.image_to_string(preprocessed_img)
        # return pytesseract.image_to_string(Image.open(image_path))


    # def extract_text_from_pdf(self, pdf_path: str) -> str:
    #     """Extract text from a PDF by converting pages to images."""
    #     images = convert_from_path(pdf_path, poppler_path=self.poppler_path)

    #     with ThreadPoolExecutor() as executor:
    #         texts = list(executor.map(self._extract_text_from_image, images))
    #     return " ".join(texts)

    # def _extract_text_from_image(self, img) -> str:
    #     """Helper method for parallel OCR processing."""
    #     return pytesseract.image_to_string(img) 
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF by converting pages to images and preprocessing."""
        images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
        
        text = ""
        for i, img in enumerate(images):
            temp_path = f"temp_page_{i}.jpg"
            img.save(temp_path, "JPEG")  # Save temp image
            preprocessed_img = self.preprocess_image(temp_path)  # Preprocess
            text += pytesseract.image_to_string(preprocessed_img) + "\n"
            os.remove(temp_path)  # Clean up temporary file
        return text
    
    # -------------------- PII Detection Methods --------------------
    def detect_structured_pii(self, text: str) -> dict:
        """Detect PII using regex (PAN, Aadhaar, email, phone, credit card)."""
        patterns = {
            "Aadhaar": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "PAN": r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",
            "Email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "Phone": r"\b(?:\+91[-\s]?)?[6789]\d{9}\b",
            "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        }
        pii_detected = {key: re.findall(pattern, text) for key, pattern in patterns.items() if re.findall(pattern, text)}
        return pii_detected

    def detect_pii_nlp(self, text: str) -> list:
        """Detect PII using Hugging Face Transformer model."""
        entities = self.hf_pii_model(text)
        pii_data = [ent["word"] for ent in entities if ent["score"] > 0.85]  # Filter by confidence
        return pii_data

    def detect_names_spacy(self, text: str) -> list:
        """Detect names using spaCy NER model."""
        doc = self.spacy_nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    def detect_names_flair(self, text: str) -> list:
        """Detect names using Flair's more accurate NER model."""
        sentence = Sentence(text)
        self.flair_tagger.predict(sentence)
        return [entity.text for entity in sentence.get_spans("ner") if entity.tag == "PER"]

    # -------------------- Redaction Methods --------------------
    def redact_text(self, text: str) -> str:
        """Mask PAN and Aadhaar numbers in text."""
        text = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]{1}\b", "XXXXX0000X", text)  # Mask PAN
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "XXXX XXXX XXXX", text)  # Mask Aadhaar
        return text

    def redact_image(self, image_path: str, output_path: str = "redacted.jpg"):
        """Draw black boxes over detected PII in original image, filtering low-confidence OCR results."""
        img = cv2.imread(image_path)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)

        for i, text in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf < 60:  # Ignore low-confidence OCR results
                continue

            structured_pii = self.detect_structured_pii(text)
            names = self.detect_pii_nlp(text)

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
