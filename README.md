# Claim Verification System

## Requirements
- Python 3.10, 3.11, or 3.12 recommended
- NVIDIA GPU with 8GB+ VRAM (RTX 5050 or equivalent)
- Ollama installed: https://ollama.ai
- 8GB free disk space

## Setup (run in order)

### 1. Install Ollama and pull model
```bash
ollama pull gemma4:e4b
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch build for your system if the default `torch` wheel does not detect CUDA.
The project uses `faiss-cpu==1.13.2` because the original `1.8.0` wheel is not available from pip for this Windows/Python environment; the retrieval code still uses FAISS `IndexFlatIP`.
For Python 3.13+, the requirements file automatically uses newer wheel-compatible NumPy, pandas, and scikit-learn versions so Windows does not try to compile them from source.
Python 3.14 is not recommended for this project because several ML/UI dependencies do not publish compatible wheels yet.
FastAPI, Starlette, Uvicorn, and Jinja are pinned because Gradio 4.44.0 is not compatible with the newest Starlette 1.x server stack.
Web retrieval uses the current `ddgs` package first; `duckduckgo-search` is kept only as a fallback because that package was renamed.

If the app prints `CUDA not available to PyTorch`, reinstall PyTorch inside the active `.venv` with the CUDA wheel:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### 3. Download NLTK data and dataset
```bash
python setup.py
```

### 4. Build search indexes (run once, takes several minutes)
```bash
python data/build_index.py
```

### 5. Launch the application
```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Running evaluation
```bash
python evaluation/evaluate.py
```

## IR Technique
This system uses a hybrid retrieval approach combining:
- BM25 (sparse/lexical) for keyword matching
- FAISS + Sentence Transformers (dense/semantic) for meaning-based search
- Reciprocal Rank Fusion for merging ranked lists
- Cross-encoder reranking for precision
- Iterative multi-hop retrieval for complex claims

Dataset: FEVER, supplemented by live web retrieval.

## GPU and Ollama Notes
Python transformer models are loaded on `cuda` when available. Ollama is never loaded through `transformers`; the app calls `gemma4:e4b` through `http://localhost:11434/api/generate`.
