```markdown
# 🚀 Enterprise Inference Engine

A headless, highly-optimized Dockerized inference microservice built with **FastAPI** and **PyTorch**. 

This project demonstrates how to take a raw Large Language Model (LLM) and deploy it as a secure, production-ready backend capable of handling high-throughput batch traffic, strict data sanitization, and mathematical output guardrails.

## 🧠 Core Engineering Features

* **Dynamic Tensor Alignment (Left-Padding):** Implements dynamic batching with explicit left-padding. This allows the PyTorch engine to process drastically uneven user prompts simultaneously as a perfect rectangular matrix, preventing the decoder-only attention mask corruption that plagues naive batching implementations.
* **$O(N)$ Memory Optimization (KV-Caching):** Enforces strict Key-Value caching during the autoregressive generation loop. This bypasses the $O(N^2)$ computational bottleneck of naive decoding, drastically reducing latency as sequence lengths scale.
* **Mathematical Guardrails (Logit Interception):** Utilizes a custom Hugging Face `LogitsProcessor` to intercept the model's forward pass before decoding. Banned vocabulary tokens are dynamically assigned a probability of $-\infty$, making it mathematically impossible for the model to output restricted enterprise terms.
* **API Gateway Security:** Built on FastAPI with strict **Pydantic** payload validation and **HTTP Bearer Token** authentication. Malformed data and unauthorized requests are dropped at the gateway, completely protecting the GPU/CPU math engine from malicious payloads.
* **Zero-Downtime State Management:** Leverages FastAPI's `lifespan` context manager to load the massive tokenizer and neural network weights into system memory exactly once during server boot, eliminating per-request cold-start delays.

## 🏗️ Architecture Stack
* **Web Framework:** FastAPI, Uvicorn, Pydantic
* **ML Engine:** PyTorch, Hugging Face Transformers, Accelerate
* **Infrastructure:** Docker
* **Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Configured for CPU execution for universal testing)

---

## ⚙️ Quick Start & Deployment

Because this system is fully containerized, it is environment-agnostic. You do not need to install Python or PyTorch on your local machine to run it.

### 1. Build the Indestructible Environment
Clone the repository and build the Docker image. The `Dockerfile` is optimized to leverage layer caching, so dependency installations are cached on subsequent builds.
```bash
docker build -t enterprise-inference-api .

```

### 2. Launch the Microservice

Start the container and map the internal API port to your local machine. You will see the lifespan engine load the neural network into memory.

```bash
docker run -p 8000:8000 enterprise-inference-api

```

---

## 📡 API Documentation

### `POST /generate`

Accepts a batch of prompts and processes them simultaneously through the inference engine.

**Authentication:** Requires a Bearer Token (`enterprise_secret_123`).
**Payload:** strict JSON array of `prompts` and a `max_tokens` limit.

#### Example Request (cURL)

Open a new terminal and ping the live server with a batch of two uneven prompts:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Authorization: Bearer enterprise_secret_123" \
     -H "Content-Type: application/json" \
     -d '{
           "prompts": [
             "Hi",
             "Explain in detail what AI model training means and why it is important."
           ],
           "max_tokens": 40
         }'

```

#### Example Response

The API correctly left-pads the "Hi" prompt, processes both concurrently, and returns a clean JSON object:

```json
{
  "status": "success",
  "results": [
    "Hi! How can I help you today?",
    "Explain in detail what AI model training means and why it is important.\n\nAI model training is the process of creating a machine learning algorithm to perform a specific task. This involves feeding the algorithm a large dataset"
  ]
}

```

#### Security Test

To verify the gateway firewall, attempt the same request with an invalid token:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Authorization: Bearer fake_token_123" \
     -H "Content-Type: application/json" \
     -d '{"prompts": ["Hi"], "max_tokens": 10}'

```

**Response:** `{"detail":"Unauthorized: Invalid Bearer Token"} (Status 401)`

```

