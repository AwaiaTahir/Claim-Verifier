# Claim Verification System

This project is an automated claim verification system built in Python. It takes in a statement, searches for relevant evidence, and outputs a verdict on whether the claim is true, false, or lacks enough information to verify.

Instead of relying on a single search method, it uses a hybrid approach. It combines standard keyword matching with semantic search to pull the most relevant context from the FEVER dataset or the live web. It then reranks the evidence and feeds it into a local language model via Ollama to determine the final verdict.

The system is designed to run entirely locally and handle complex claims that require multiple steps of reasoning to verify.

## System Requirements

* Python 3.10, 3.11, or 3.12 (Recommended)
* NVIDIA GPU with 8GB+ VRAM (RTX 5050 or equivalent)
* Ollama installed and running (https://ollama.ai)
* 8GB of free disk space

## Installation and Setup

Please follow these steps in order to ensure the system runs correctly.

### 1. Install Ollama and Pull the Model

Ensure Ollama is installed on your system. Pull the required model by running:

```bash
ollama pull gemma4:e4b
```

### 2. Install Python Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

If the application prints a message stating that CUDA is not available to PyTorch, you will need to reinstall PyTorch inside your active virtual environment using the CUDA wheel:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

You can verify the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### 3. Download Required Data

Download the NLTK data and required datasets:

```bash
python setup.py
```

### 4. Build Search Indexes

This step only needs to be run once and may take several minutes to complete.

```bash
python data/build_index.py
```

### 5. Launch the Application

Start the local server and web interface:

```bash
python app.py
```

Once the server is running, open http://localhost:7860 in your web browser to use the application.

## Running the Evaluation

To evaluate the system against the dataset, run the evaluation script:

```bash
python evaluation/evaluate.py
```

## Information Retrieval Technique

This system uses a hybrid retrieval approach combining the following methods:

* BM25 for sparse/lexical keyword matching
* FAISS and Sentence Transformers for dense/semantic meaning-based search
* Reciprocal Rank Fusion for merging ranked lists
* Cross-encoder reranking for improved precision
* Iterative multi-hop retrieval for complex claims

The system utilizes the FEVER dataset, supplemented by live web retrieval when necessary.

## GPU and Model Notes

Python transformer models are loaded onto the GPU via CUDA when available. The Ollama model is not loaded through the transformers library; instead, the application calls the local model through its API endpoint (`http://localhost:11434/api/generate`).

Compatibility notes regarding dependencies:

* The project uses `faiss-cpu==1.13.2` as the `1.8.0` wheel is unavailable for this environment. The retrieval code utilizes FAISS `IndexFlatIP`.
* Python 3.13+ environments will automatically use wheel-compatible versions of NumPy, pandas, and scikit-learn.
* Python 3.14 is not recommended at this time due to missing compatible wheels for several ML and UI dependencies.
* FastAPI, Starlette, Uvicorn, and Jinja are pinned to specific versions to maintain compatibility with Gradio 4.44.0.
* Web retrieval uses the current `ddgs` package first; `duckduckgo-search` is kept only as a fallback.
