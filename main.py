from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# --- 1. Environment & Global State ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SECURITY_TOKEN = "enterprise_secret_123" 
BANNED_WORDS = ["competitor", "hack", "illegal"]

# Global dictionary to hold the model in server memory
ml_env = {}

# --- 2. Lifespan Engine Loading ---
# We load the heavy ML models exactly once during server startup, not per-request
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ SERVER BOOT: Loading Tokenizer and weights into memory...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Ensure pad_token is set for rectangular batch alignment
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Loading to CPU (or "cuda" if hardware is available)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")
    
    ml_env["tokenizer"] = tokenizer
    ml_env["model"] = model
    
    print("✅ SERVER BOOT: Inference Engine Online.")
    yield
    # Cleanup memory on shutdown
    ml_env.clear()

# --- 3. API Setup & Security ---
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != SECURITY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Bearer Token")
    return credentials.credentials

# --- 4. Strict Data Validation (Pydantic) ---
class InferenceRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 40

class InferenceResponse(BaseModel):
    status: str
    results: list[str]

# --- 5. Custom Mathematical Guardrail ---
class GuardrailLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_ids: list[int]):
        self.banned_ids = banned_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Intercept the forward pass logits and crush the probability of banned tokens
        for banned_id in self.banned_ids:
            scores[:, banned_id] = -float('inf')
        return scores

# --- 6. The Core Inference Endpoint ---
@app.post("/generate", response_model=InferenceResponse, dependencies=[Depends(verify_token)])
async def generate_text(payload: InferenceRequest):
    tokenizer = ml_env["tokenizer"]
    model = ml_env["model"]
    
    # Preprocessing: Batching & Padding
    # Automatically pads sequences of varying lengths with zeros to create a perfect rectangular tensor[cite: 1]
    inputs = tokenizer(payload.prompts, padding=True, return_tensors="pt")
    
    # Find the numerical vocabulary IDs for our banned words
    banned_ids = []
    for word in BANNED_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        banned_ids.extend(ids)
        
    processors = LogitsProcessorList([GuardrailLogitsProcessor(banned_ids)])
    
    print(f"🚦 PROCESSING BATCH: {len(payload.prompts)} parallel inputs...")
    
    # Forward Pass & Autoregressive Decoding[cite: 1]
    with torch.inference_mode():
        # use_cache=True enforces KV-Caching to avoid recomputing historical states at every step[cite: 1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=payload.max_tokens,
            use_cache=True, 
            logits_processor=processors
        )
    
    # Postprocessing: Detokenize the raw numerical tensor back into human-readable text[cite: 1]
    decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return InferenceResponse(
        status="success",
        results=decoded_responses
    )