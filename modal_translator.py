import modal
from modal import Image, Stub, web_endpoint
import time
from typing import Dict, Optional
import torch

stub = Stub("medical-translator-pro")

# Define container image
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers==4.35.2",
        "torch==2.1.0",
        "sentencepiece==0.1.99",
        "protobuf==4.24.4",
        "accelerate==0.25.0",
        "langdetect==1.0.9",
    )
)

@stub.cls(
    image=image,
    gpu="A10G",
    memory=32768,
    container_idle_timeout=600,
)
class MedicalTranslator:
    def __enter__(self):
        """Initialize model on container startup"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from langdetect import detect_langs
        
        print("🚀 Loading Medical Translator...")
        
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📊 Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)
        
        self.model.eval()
        
        # Language mappings
        self.lang_mapping = {
            'es': 'spa_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'it': 'ita_Latn', 'pt': 'por_Latn', 'ru': 'rus_Cyrl',
            'ar': 'ara_Arab', 'zh-cn': 'zho_Hans', 'ja': 'jpn_Jpan',
            'ko': 'kor_Hang', 'hi': 'hin_Deva', 'tr': 'tur_Latn'
        }
        
        self.lang_names = {
            'spa_Latn': 'Spanish', 'fra_Latn': 'French', 'deu_Latn': 'German',
            'ita_Latn': 'Italian', 'por_Latn': 'Portuguese', 'rus_Cyrl': 'Russian',
            'ara_Arab': 'Arabic', 'zho_Hans': 'Chinese', 'jpn_Jpan': 'Japanese',
            'kor_Hang': 'Korean', 'hin_Deva': 'Hindi', 'tur_Latn': 'Turkish'
        }
        
        print("✅ Model loaded successfully!")
    
    def detect_language(self, text: str) -> tuple:
        """Detect language of input text"""
        try:
            from langdetect import detect_langs
            detections = detect_langs(text)
            if detections:
                lang = detections[0]
                lang_code = str(lang.lang)
                confidence = lang.prob
                nllb_code = self.lang_mapping.get(lang_code, 'spa_Latn')
                return nllb_code, confidence
        except:
            pass
        
        # Fallback detection
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'zho_Hans', 0.9
        elif any('\u0600' <= c <= '\u06ff' for c in text):
            return 'ara_Arab', 0.9
        elif any('\u3040' <= c <= '\u309f' for c in text):
            return 'jpn_Jpan', 0.9
        elif any('\uac00' <= c <= '\ud7af' for c in text):
            return 'kor_Hang', 0.9
        
        return 'spa_Latn', 0.7
    
    @modal.method()
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "eng_Latn",
        verification_passes: int = 3
    ) -> Dict:
        """Translate text to English"""
        start_time = time.time()
        
        # Auto-detect language
        if not source_lang:
            source_lang, confidence = self.detect_language(text)
        else:
            confidence = 1.0
        
        lang_name = self.lang_names.get(source_lang, 'Unknown')
        
        # Tokenize
        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Translate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                max_length=1024,
                num_beams=5,
                temperature=0.8,
                early_stopping=True
            )
        
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "translated_text": translated_text,
            "detected_language": source_lang,
            "detected_language_name": lang_name,
            "confidence_score": float(confidence),
            "processing_time": float(time.time() - start_time),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    
    @modal.web_endpoint(method="POST")
    def translate_api(self, request: Dict) -> Dict:
        """Web API endpoint"""
        text = request.get("text", "").strip()
        if not text:
            return {"error": "No text provided"}
        
        return self.translate(text=text)

# Health check endpoint
@stub.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {
        "status": "healthy",
        "service": "medical-translator-pro",
        "gpu_available": torch.cuda.is_available()
    }
