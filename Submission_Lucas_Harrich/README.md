# Project Report

This project compares four setups for Austrian tax question answering. The test set comes from `Code/dataset_clean.csv` and contains 643 prompts. I generated one answer file for each setup and then compared the answers with the gold solutions from `Evaluation/solutions.csv`.

## Models And Data

The first setup was `Qwen/Qwen2.5-1.5B-Instruct`. This is a 1.5B parameter instruct model. In the notebook `Code/M1_inference_qwen_1_5b.ipynb` I used `maxInputTokens = 420`, `maxNewTokens = 160`, `do_sample = False`, and `repetition_penalty = 1.05`. This means the model answered in a deterministic way and did not use sampling. According to the official Qwen2.5 release, Qwen2.5 models were pretrained on up to 18 trillion tokens. The pretraining data is a large multilingual mix with web text and stronger code and math coverage. The model family supports more than 29 languages. The instruct version is the chat aligned version of that base model.

The second setup was a fine tuned version of the same base model. I used `Qwen/Qwen2.5-1.5B-Instruct` again, but this time with supervised fine tuning on my own small seed dataset in `Code/finetune_data/austrian_corp_tax_seed_sft.jsonl`. This dataset contains 146 examples and focuses on Austrian corporate tax law. The sources are mainly official Austrian legal sources, especially RIS, and also Findok and BMF material. In `Code/M2_finetune_qwen_1_5b.ipynb` I converted the data to MLX LM format and trained with `mlx_lm` in full fine tune mode on the last two layers. The main settings were `trainPasses = 2`, `iters = 292`, `batch size = 1`, `gradient accumulation = 4`, `learning rate = 2e-6`, `max sequence length = 300`, and `seed = 42`. For generation after training I used `maxNewTokens = 130`, `temperature = 0.7`, `top_p = 0.9`, `top_k = 40`, and `repetition_penalty = 1.08`.

The third setup was a RAG system. The retrieval model was `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which creates 384 dimensional sentence embeddings. The Sentence Transformers documentation describes it as a multilingual version of `paraphrase-MiniLM-L12-v2`, trained on parallel data for more than 50 languages. The original paraphrase model family was trained for semantic similarity and paraphrase style tasks. For the document index I used `Code/rag_data/ris_kstg_qa_corpus.jsonl`, which contains 138 entries. These entries are short question answer style passages based on official RIS material for Austrian corporate tax law. The preprocessing was simple. I did not split large documents into many overlapping blocks. Instead, each legal fact or question answer pair became one short chunk, and each chunk also got a `search_text` field for retrieval. In `Code/M3_RAG_qwen_1_5b.ipynb` I retrieved the top 5 passages for each question with a hybrid score from dense similarity and lexical overlap. The generation model was `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. I used `maxNewTokens = 80`, `temperature = 0.1`, `top_p = 0.9`, `top_k = 20`, `repetition_penalty = 1.10`, and `directAnswerThreshold = 0.72`. In the saved notebook run, the system answered 14 cases by direct retrieval, 261 cases by retrieval fallback, and 368 cases with the generator output.

The fourth setup was `Qwen/Qwen3-14B`. This is a much larger dense model with about 14B parameters. In `Code/M4_Additional_model_qwen_14b.ipynb` I used `enableThinking = False`, `maxInputTokens = 600`, `maxNewTokens = 180`, `do_sample = False`, and `repetition_penalty = 1.05`. So this run also used deterministic generation, but with a much stronger model. According to the official Qwen3 release, Qwen3 was pretrained on about 36 trillion tokens across 119 languages and dialects. The data includes web text, PDF like documents, and synthetic data for code and math such as textbooks, question answer pairs, and code snippets.

## Evaluation

I used BERTScore F1 as the main automatic evaluation metric. The summary file is `Evaluation/bertscore_summary.csv`. All prediction files contain 643 answers, but only 641 rows were scored because two IDs from the prediction files, `VAT-INTL-081` and `VAT-INTL-082`, do not appear in the gold file, while the gold file contains `ESTG27-015` and `ESTG27-016` instead. So the real comparison was done on the 641 overlapping questions.

The main result table is shown below.

| Model | Overall BERTScore F1 | CORP TAX BERTScore F1 | Rows Scored |
| --- | ---: | ---: | ---: |
| Qwen3 14B | 0.7078 | 0.6887 | 641 |
| Qwen2.5 1.5B fine tuned | 0.6857 | 0.6706 | 641 |
| Qwen2.5 1.5B base | 0.6845 | 0.6543 | 641 |
| Qwen2.5 1.5B RAG | 0.6784 | 0.7030 | 641 |

The best model on the full benchmark was `Qwen/Qwen3-14B`. It had the highest overall BERTScore F1 by a clear margin. The fine tuned 1.5B model was only a little better than the base 1.5B model. This makes sense because the fine tuning dataset is small and only covers one part of the full benchmark. The RAG system was not the best overall, but it was strongest on the `CORP-TAX` subset. This also makes sense because the indexed corpus only covers Austrian corporate tax material from RIS. So the RAG setup helped in its own narrow domain, but it lost too much on the broader tax questions in the full test set.

## Error Analysis

The most common error across all models was wrong legal focus. Some answers sounded fluent, but they moved into the wrong tax area or used generic tax language instead of the exact legal rule. A good example is `GRESt-AT-063`. Here several models moved away from the actual real estate transfer tax question, and the 14B model even answered with inheritance tax logic. Another common issue was incomplete legal reasoning. In `VAT-INTL-069` some models gave the right country, Austria, but the legal explanation was still wrong or too generic. In `SELF-061` the models often answered with broad bookkeeping language, but they missed the exact formal requirements from `§ 131 BAO`.

The mistakes were not fully the same for all setups. The smaller base and fine tuned models often produced long and vague answers with invented details. The fine tuned model was a bit more focused in the corporate tax area, but outside that area it had almost no advantage. The RAG system had a different failure mode. When retrieval worked, it was precise and safe, especially for corporate tax. When retrieval missed the topic, the answer could become unrelated because the indexed corpus was too narrow. This can be seen in some VAT and self employment questions where the model returned corporate tax style content. The 14B model was the strongest overall, but when it failed, the answer was often very confident and polished, which makes the error harder to notice.

## Short Conclusion

For the full task, the larger general model was the best choice. `Qwen/Qwen3-14B` gave the strongest overall result. The small fine tuned model only improved a little because the fine tuning data was small and narrow. The RAG setup was useful for Austrian corporate tax questions, but it needs a much broader index if it should work well on the full benchmark.