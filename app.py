import modal

app = modal.App("medical-translator")

@app.cls(
    image=modal.Image.debian_slim().pip_install(
        "transformers",
        "torch", 
        "sentencepiece",
        "accelerate",
        "fastapi"  # ← THIS WAS MISSING!
    ),
    gpu=modal.gpu.T4(),
    container_idle_timeout=300,
)
class Translator:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        print("Loading NLLB-200 translation model...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            torch_dtype=torch.float16
        )
        self.model = self.model.to("cuda")
        print("Model ready!")
    
    @modal.web_endpoint(method="POST")
    def translate(self, request: dict):
        text = request.get("text", "")
        
        # Auto-detect language
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            src_lang = "zho_Hans"  # Chinese
        elif any('\u0600' <= c <= '\u06ff' for c in text):
            src_lang = "ara_Arab"  # Arabic
        elif any('\u3040' <= c <= '\u309f' for c in text):
            src_lang = "jpn_Jpan"  # Japanese
        else:
            src_lang = "spa_Latn"  # Default Spanish
        
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        
        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"],
            max_length=1000
        )
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return {
            "translation": translation,
            "detected_language": src_lang,
            "status": "success"
        }
    
    @modal.web_endpoint(method="GET")
    def health(self):
        return {"status": "healthy", "model": "NLLB-200 (600M params)"}
