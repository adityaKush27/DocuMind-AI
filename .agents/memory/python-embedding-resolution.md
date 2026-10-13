---
name: Python embedding package resolution
description: Replit Python package-index behavior for local sentence embedding libraries.
---

When the Replit Python package manager cannot resolve `sentence-transformers`, verify its configured index before retrying versions. In this workspace, the package manager routed it to the PyTorch CPU index, which has no package releases. FastEmbed was available from the configured Python package index and can run the same `sentence-transformers/all-MiniLM-L6-v2` embedding model through ONNX.

**Why:** the latest and a pinned Sentence Transformers install both failed because of the package-index mapping, while FastEmbed installed and successfully produced the expected 384-dimensional embeddings.

**How to apply:** for a local MiniLM semantic-search pipeline, prefer FastEmbed when the package manager reproduces this index-resolution failure; preserve the same model and cosine-search behavior.
