"""Download the embedding model, reranker and LLM into one folder (used when building the API image).

python scripts/fetch_models.py /app/models
"""

import json
import sys
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import hf_hub_download

out = Path(sys.argv[1])
meta = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "serving/index/meta.json").read_text())
TextEmbedding(meta["embedder"], cache_dir=str(out))
TextCrossEncoder(meta["reranker"], cache_dir=str(out))
gguf = hf_hub_download(repo_id=meta["llm_repo"], filename=meta["llm_file"], local_dir=str(out / "llm"))
print("models ready:", gguf)
