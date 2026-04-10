# VAT-INTL-001 — Austrian Tax Law Q&A Models

Three models for answering Austrian tax law questions (KStG, EStG, UStG), evaluated on 644 test questions from `dataset_clean.csv`.

## Models

### Model 1: Local Inference (`model1_inference.ipynb`)

Pure inference using `dbmdz/german-gpt2` — a GPT-2 model pre-trained on German text. No fine-tuning, no API. The model receives each question formatted as `"Frage: ...\nAntwort:"` and generates a continuation using greedy decoding. Since the model was never trained on tax law, this serves as a **baseline** to show what a general-purpose German model produces without any domain adaptation.

- **Run locally** (VS Code, Jupyter, or Colab)
- No API key needed
- ~20-30 minutes on CPU

### Model 2: Fine-tuned Model (`model2_finetune.ipynb`)

Fine-tunes `dbmdz/german-gpt2` on 152 Austrian tax law Q&A pairs, then runs inference on all 644 test questions. The training data (`training_data.csv`) was written manually from the actual law texts — KStG 1988, EStG 1988, and UStG 1994.

The fine-tuning uses HuggingFace `Trainer` with causal language modeling (next-token prediction on `"Frage: ...\nAntwort: ..."` formatted text). After training for 3 epochs, the model learns the Q&A format and some domain-specific patterns from the tax law content.

- **Run on Google Colab with T4 GPU** (Runtime > Change runtime type > T4 GPU)
- No API key needed
- Upload `training_data.csv` to Colab before running, or uncomment the GitHub URL line in the notebook
- ~5-10 minutes training on T4 GPU

### Model 3: RAG (`model3_rag.ipynb`)

Retrieval-Augmented Generation using the actual Austrian tax law PDFs as source documents. For each question: (1) extract text from the 3 law PDFs, (2) split by paragraph markers, (3) embed all chunks using OpenAI `text-embedding-3-small`, (4) find the 3 most relevant chunks via cosine similarity, (5) send the chunks + question to `gpt-4o-mini` for answer generation.

This is the **only model that uses the OpenAI API** (for embeddings and generation). It's expected to perform best because the model has the actual law text in its context at query time.

- **Run locally** (VS Code or Jupyter)
- **Requires an OpenAI API key** — set `API_KEY` in the notebook before running
- The 3 PDF files must be present in `../../Context/Gesetze/`
- ~15-20 minutes, costs ~$0.50 in API calls

## API Usage

Per the course requirements, only **one model may use an external API**. In this project:

| Model | API used? | What runs locally |
|-------|-----------|-------------------|
| Model 1 | No | dbmdz/german-gpt2 inference via HuggingFace |
| Model 2 | No | dbmdz/german-gpt2 fine-tuning via HuggingFace Trainer |
| Model 3 | **Yes** | PDF extraction + chunking are local; embeddings + generation use OpenAI |

**Note on Model 2 training data:** The 152 Q&A pairs in `training_data.csv` were written by AI from the law texts. An earlier version of this notebook used the OpenAI API to generate synthetic training data, but this was replaced to comply with the one-API-model rule. The training data is committed to the repository and can be inspected directly.

## How to Reproduce

1. Clone the repository
2. For Model 3: get an OpenAI API key from https://platform.openai.com/api-keys and paste it into the notebook
3. Run `model1_inference.ipynb` locally
4. Upload `model2_finetune.ipynb` + `training_data.csv` to Google Colab, enable T4 GPU, run all cells, download `model2_results.csv`
5. Run `model3_rag.ipynb` locally (ensure the 3 law PDFs are in `../../Context/Gesetze/`)
6. All results are saved in `results/`

## File Structure

```
VAT-INTL-001/
  code/
    model1_inference.ipynb   # Local inference (no API)
    model2_finetune.ipynb    # Fine-tuning on Colab (no API)
    model3_rag.ipynb         # RAG with OpenAI API
    training_data.csv        # 152 Q&A pairs for Model 2
  results/
    model1_results.csv
    model2_results.csv
    model3_results.csv
```
