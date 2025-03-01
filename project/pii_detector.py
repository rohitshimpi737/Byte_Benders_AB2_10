import re
import pytesseract
from PIL import Image
import spacy
from pdf2image import convert_from_path  # Add this import
from flair.models import SequenceTagger  # Add these imports
from flair.data import Sentence
from concurrent.futures import ThreadPoolExecutor
from pytesseract import Output  # Missing in your code



class PIIDetector:
    def __init__(self, tesseract_cmd_path: str):
        self.nlp = spacy.load("en_core_web_sm")  # Load NLP model
        self.flair_tagger = SequenceTagger.load("ner")  # Load Flair model
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path  # Use the provided Tesseract path
        
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        return pytesseract.image_to_string(Image.open(image_path))
    
    def detect_names_flair(self, text: str) -> list:
        """Detect names using Flair's NER (more accurate)."""
        sentence = Sentence(text)
        self.flair_tagger.predict(sentence)
        return [entity.text for entity in sentence.get_spans("ner") if entity.tag == "PER"]
    
    # def detect_structured_pii(self, text: str) -> dict:
    #     """Detect PAN, Aadhaar, and emails using regex."""
    #     pii = {
    #         "Aadhaar": re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", text),
    #         "PAN": re.findall(r"[A-Z]{5}\d{4}[A-Z]{1}", text),
    #         "Email": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    #     }
    #     return pii

    

    def detect_contextual_pii(self, text: str) -> list:
        """Detect sentences mentioning PII keywords (e.g., 'My Aadhaar is...')."""
        doc = self.nlp(text)
        pii_keywords = ["aadhaar", "pan", "email", "phone"]
        contextual_pii = []
    
        for sent in doc.sents:
            for token in sent:
                if token.text.lower() in pii_keywords and token.head.dep_ == "attr":
                    contextual_pii.append(sent.text)
    
        return contextual_pii

    def check_compliance(self, pii_data: dict, consent_given: bool = False) -> list:
        """Flag PII requiring explicit consent."""
        non_compliant = []
        if pii_data.get("Aadhaar") and not consent_given:
            non_compliant.append("Aadhaar requires consent under DPDP 2023.")
        return non_compliant


    def detect_structured_pii(self, text: str) -> dict:
        pii = {
            # Existing patterns
            "Aadhaar": re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", text),
            "PAN": re.findall(r"[A-Z]{5}\d{4}[A-Z]{1}", text),
            "Email": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text),
            # New patterns
            "Phone": re.findall(r"\b(?:\+91|0)?[6-9]\d{9}\b", text),  # Indian numbers
            "Address": re.findall(r"\b\d{1,5}\s[\w\s]{5,}\b", text)  # Simple street addresses
        }
        return pii
    
    
    def detect_names(self, text: str) -> list:
        """Detect names using spaCy's NER model."""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    def redact_text(self, text: str) -> str:
        """Mask PAN and Aadhaar numbers."""
        # Mask PAN (e.g., ABCDE1234F → ABCDE****F)
        redacted = re.sub(r"([A-Z]{5})(\d{4})([A-Z]{1})", r"\1****\3", text)
        # Mask Aadhaar (e.g., 1234 5678 9101 → **** **** ****)
        redacted = re.sub(r"\d{4}\s\d{4}\s\d{4}", "**** **** ****", redacted)
        return redacted
    
    # def extract_text_from_pdf(self, pdf_path: str) -> str:
    #     """Extract text from PDF by converting pages to images."""
    #     images = convert_from_path(pdf_path)
    #     text = ""
    #     for img in images:
    #         text += pytesseract.image_to_string(img)
    #     return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        images = convert_from_path(pdf_path)
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(self.extract_text_from_image, images))
        return " ".join(texts)

    def extract_text_from_image(self, img):
        return pytesseract.image_to_string(img)
    
    def redact_image(self, image_path: str, output_path: str = "redacted.jpg"):
        """Draw black boxes over PII in the original image."""
        img = cv2.imread(image_path)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        for i, text in enumerate(data["text"]):
            if self.detect_structured_pii(text) or self.detect_names(text):
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Black box
        
        cv2.imwrite(output_path, img)
    