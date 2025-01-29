import torch
from transformers import AutoTokenizer
from model import SmolLM2ForCausalLM, SmolLM2Config
import os
import json

class SmolLM2Generator:
    def __init__(self, model_path: str = "smollm2_model_final"):
        """Initialize the model and tokenizer"""
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        self.model, self.tokenizer = self._load_model(model_path)
        print("Model loaded successfully!")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _load_model(self, model_path: str):
        """Load model and tokenizer from saved files"""
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SmolLM2Config(**config_dict)
        
        # Initialize and load model
        model = SmolLM2ForCausalLM(config)
        model_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(
            model_path, 
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        **kwargs
    ) -> str:
        """Generate text with advanced parameters"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = []
        current_length = input_ids.size(1)
        
        with torch.no_grad():
            while current_length < max_length:
                outputs, _ = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token in generated_tokens:
                        next_token_logits[0, token] /= repetition_penalty
                
                # Filter special tokens
                for special_token_id in [self.tokenizer.pad_token_id, 
                                       self.tokenizer.eos_token_id, 
                                       self.tokenizer.bos_token_id]:
                    if special_token_id is not None:
                        next_token_logits[0, special_token_id] = float('-inf')
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs[0], num_samples=1)
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                current_length += 1
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text 