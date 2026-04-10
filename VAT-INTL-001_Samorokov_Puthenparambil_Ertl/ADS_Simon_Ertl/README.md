# Applications of Data Science - Summer Term 2026
## Project: Building an Austrian Tax Law Assistant

**Author**:  Simon Andreas ERTL
---
This folder contains my submission files for task 2 of the project.
This folder contains three Jupyter Notebooks, each containing one model. **All notebooks are supposed to be run within Google Colab. Every notebook is compatible with the T4 architecture available in the free tier.**

`model1_inference_ERTL.ipynb` can be run on alone. It does not depend on the other two notebooks. It uses a out-of-box LLM to answer the questions about Austrian Tax Law. This notebook takes the .csv-file provided as input and outputs a .csv-file with the question IDs and the answers.

`model2_sft_ERTL.ipynb` fine-tunes the model using `sft_dataset.json`.  This model takes `dataset_clean.csv` (for question answering) and `sft_dataset.json` (for fine-tuning) as input. It outputs a .csv-file with the results and a .zip-archive with the LoRA adapter.

`model3_RAG_ERTL.ipynb` applies RAG to the fine-tuned model. It takes `dataset_clean.csv` (for question answering), the folder `model2_lora_adapter` (with the files within), and the folder `context` (with the PDF files within) as an input and outputs a .csv-file.

Outputs are directly written to the project root.
