# DocuMind AI — Smart Document Q&A Assistant

DocuMind AI is a session-based Streamlit app for uploading PDFs, searching their text by meaning, asking Gemini questions grounded in the documents, and creating summaries. It is intentionally small enough to explain in a software engineering interview.

## Features

- Upload one or more text-based PDF files.
- Extract page text with PyPDF and split it into overlapping passages.
- Encode passages locally with the `all-MiniLM-L6-v2` Sentence Transformers model (served by the lightweight FastEmbed ONNX runtime) and NumPy cosine similarity.
- Retrieve up to three relevant passages for each question.
- Ask Gemini for an answer grounded in those passages, with PDF and page references.
- Summarize large documents a section at a time.
- Keep documents, embeddings, summaries, and chat in Streamlit session state; nothing is stored in a database.
- Report missing keys, invalid or encrypted PDFs, scanned PDFs without selectable text, and provider errors without inventing answers.

## Run in Replit

1. Add a Replit Secret named `GEMINI_API_KEY` with your Google Gemini API key. The secret is read by the server and is never shown in the interface.
2. Start the **DocuMind AI** workflow.
3. Upload a PDF. The first upload downloads the local sentence embedding model; later uploads reuse it.

If the secret is absent, the app remains available for PDF processing and clearly disables AI answers and summaries until the key is configured.

## Run locally

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY to your Google AI Studio key.
streamlit run app.py --server.address 0.0.0.0 --server.port 5000
```

Never commit `.env`. For Replit, use **Tools → Secrets** instead of writing a key into a file.

## Project structure

- `app.py` — Streamlit layout, upload lifecycle, session state, chat, summaries, and error states.
- `pdf_processor.py` — PDF validation, page-level text extraction, and overlapping chunk creation.
- `rag_pipeline.py` — local embeddings, cosine retrieval, Gemini requests, and section-based summaries.
- `requirements.txt` — Python packages.
- `.env.example` — safe local configuration example with a placeholder only.

## RAG flow

1. PyPDF extracts selectable text from each page.
2. `pdf_processor.py` creates approximately 500-character passages with an 80-character overlap and preserves source page numbers.
3. FastEmbed runs the `sentence-transformers/all-MiniLM-L6-v2` model locally and creates normalized embeddings, cached in memory for the current Streamlit server process.
4. The question is embedded and compared with stored vectors using cosine similarity.
5. Up to three relevant passages are sent to `gemini-3.8-flash` with instructions to stay within the supplied context. Set `GEMINI_MODEL` to override the default if your account uses a different supported model.
6. The UI displays the answer alongside the source PDF, page numbers, and expandable retrieved context.

The local embedding model does not require a Gemini key. Gemini does; without `GEMINI_API_KEY`, no AI answer or summary is fabricated.

## Notes and limits

- Each PDF is limited to 30 MB.
- Documents and conversation are kept in Streamlit session state and disappear when the session is cleared or the server restarts.
- Image-only scans need OCR and are reported clearly rather than treated as empty answers.
- Gemini requests use a small retry policy for temporary service and quota errors.
