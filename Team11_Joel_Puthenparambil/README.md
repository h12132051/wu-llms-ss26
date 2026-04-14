# Team11 — Joel Puthenparambil
## WU LLM Course SS26 — Austrian Tax Law Q&A

**Task:** Given a German-language Austrian tax law question, generate a 1–4 sentence answer in German citing the relevant statute (e.g. `§ 7 Abs 1 KStG`).

**Dataset:** `dataset_clean.csv` — 643 questions, used only for inference (test set). No ground truth answers are included.

---

## Models

### Model 1 — Inference Only

| Property | Value |
|----------|-------|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantization | 4-bit (bitsandbytes) |
| Platform | Google Colab (free T4 GPU) |
| Notebook | `code/model1_inference.ipynb` |
| Output | `results/model1_inference.csv` |

**Approach:** The model is loaded in 4-bit quantization via `BitsAndBytesConfig` and prompted with a shared system prompt instructing it to answer in German citing Austrian statutes. No fine-tuning or retrieval — pure zero-shot inference.

**System prompt (all models):**
```
Beantworte die folgende Frage zum österreichischen Steuerrecht auf Deutsch.
Antworte in maximal 1–4 Sätzen.
Nenne die einschlägige Rechtsnorm (z.B. § 7 Abs 1 KStG).
Halluziniere keine Paragraphen. Wenn unklar, formuliere vorsichtig.
```

---

### Model 2 — Fine-Tuned

| Property | Value |
|----------|-------|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Method | LoRA fine-tuning (PEFT), fp16, no bitsandbytes |
| LoRA rank | r=16, alpha=32, targets: q_proj / v_proj |
| Epochs | 3 |
| Training data | 758 Q&A pairs generated from scraped RIS statute sections |
| Platform | Lightning.ai (free T4 GPU) |
| Notebook | `code/model2_finetune.ipynb` |
| Output | `results/model2_finetuned.csv` |

**Training data:** Template-generated question-answer pairs from Austrian statute text scraped from RIS (ris.bka.gv.at). Statutes covered: EStG 1988, KStG 1988, UStG 1994, GrEStG 1987, BAO. The `dataset_clean.csv` test set was NOT used for training.

**Key technical decisions:**
- Used `SFTConfig` from `trl` (not `TrainingArguments`) — required by newer trl versions
- Before inference: `model.gradient_checkpointing_disable()` + `model.config.use_cache = True` + `model.eval()`
- Inference uses `repetition_penalty=1.3`, `do_sample=True`, `temperature=0.5` to prevent token repetition loops

---

### Model 3 — RAG (Retrieval-Augmented Generation)

| Property | Value |
|----------|-------|
| Retriever | `paraphrase-multilingual-mpnet-base-v2` + FAISS IndexFlatIP |
| Top-K | 3 statute passages |
| Generator | `gemini-2.5-flash-lite` via `google-genai` SDK |
| RAG corpus | EStG 1988, KStG 1988, UStG 1994, GrEStG 1987, BAO (RIS HTML scraping) |
| Platform | Google Colab (CPU) |
| Notebook | `code/model3_rag.ipynb` |
| Output | `results/model3_rag.csv` |

**Approach:**
1. Scrape Austrian statute text from RIS (ris.bka.gv.at) by law ID
2. Split into ~500-character passages
3. Embed all passages with `paraphrase-multilingual-mpnet-base-v2`
4. At inference time: embed the question, retrieve top-3 passages via cosine similarity (FAISS)
5. Pass question + retrieved passages to Gemini 2.5 Flash-Lite for generation

**Rate limiting:** `BATCH_SIZE=10`, 65-second pause between batches to stay under 15 RPM. Billing enabled (~$0.07 total).

---

## Evaluation

### Methodology

Reference answers come from the course dataset Google Sheet, which contains golden-standard answers created by the course team. All three model outputs are evaluated against these using **BERTScore** (`xlm-roberta-base`).

BERTScore measures semantic similarity using contextual embeddings, making it well-suited for German legal text where paraphrasing is common and exact n-gram matches are rare.

Evaluation notebook: `code/evaluate.ipynb`

### Results

All three models evaluated against the golden-standard answers from the course dataset using BERTScore (`xlm-roberta-base`, German).

| Model | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|-------|---------------------|------------------|--------------|
| Model 1 (Mistral-7B inference) | 0.8372 | 0.8544 | 0.8454 |
| Model 2 (TinyLlama fine-tuned) | 0.8306 | 0.8193 | 0.8246 |
| Model 3 (RAG + Gemini) | 0.8515 | 0.8688 | **0.8597** |

Full results saved in `results/evaluation_report.csv`.

---

## Error Analysis

**Model 1 (Mistral-7B):** Generally produces grammatically correct German with plausible statute citations. Occasionally cites paragraphs that don't exist or conflates §-numbers across laws (hallucination). Answers tend to be verbose.

**Model 2 (TinyLlama fine-tuned):** The 1.1B parameter base model is small for this domain. Answers sometimes drift off-topic or repeat phrases. Fine-tuning on 758 template-generated pairs improved citation format consistency but did not fully cure hallucination. The main quality bottleneck is model capacity, not the LoRA configuration.

**Model 3 (RAG + Gemini):** Highest answer quality. Retrieved statute passages ground the generator in real legal text, reducing hallucination. Occasional failures occur when the relevant statute is not among the scraped laws (e.g., niche VAT sub-regulations). Gemini's 2.5 Flash-Lite generation is fast and coherent.

---

## Repository Structure

```
Team11_Joel_Puthenparambil/
├── code/
│   ├── model1_inference.ipynb   # Mistral-7B 4-bit inference
│   ├── model2_finetune.ipynb    # TinyLlama LoRA fine-tuning + inference
│   ├── model3_rag.ipynb         # FAISS retrieval + Gemini generation
│   └── evaluate.ipynb           # ROUGE / BERTScore evaluation
└── results/
    ├── model1_inference.csv     # 643 rows
    ├── model2_finetuned.csv     # 643 rows
    └── model3_rag.csv           # 643 rows
```

---

## Reproduction

All notebooks are self-contained and run on Google Colab free tier (T4 GPU for Models 1 and 2, CPU for Model 3). Upload `dataset_clean.csv` when prompted.

**Model 3 only:** requires a Google AI Studio API key with billing enabled (free tier: 20 requests/day, insufficient for 643 questions).
