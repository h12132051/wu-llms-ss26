# Austrian Tax LLM Project - Luca Rigel

This folder contains the implementation and results for the Austrian Tax LLM assignment (SS26).

## 📁 Project Structure
- **Luca_Rigel/code/**: 
  - `model_1_inference.py`: Zero-shot answering using `flan-t5-base`.
  - `model_2_finetuning.py`: `distilgpt2` fine-tuned on tax textbooks.
  - `model_3_rag.py`: RAG pipeline using context retrieval from legal documents.
- **Luca_Rigel/results/**:
  - `model_1_output.csv`
  - `model_2_output.csv`
  - `model_3_output.csv`

## 🤖 Models Implementation
1. **Inference Only**: Baseline results using the `flan-t5-base` model without additional context.
2. **Fine-Tuning**: A `distilgpt2` model trained specifically on the provided Austrian tax PDF textbooks.
3. **RAG (Retrieval-Augmented Generation)**: Implementation that searches for relevant legal passages from PDFs and passes them to the LLM for context-aware generation.

## 🛠 Technical Notes
- **Hardware**: Optimized for local execution with Apple Silicon GPU acceleration (MPS).
- **Format**: All submissions follow the required `id,answer` CSV format with 644 rows.
