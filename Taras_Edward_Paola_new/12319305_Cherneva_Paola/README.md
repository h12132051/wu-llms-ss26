# Austrian Tax Law LLM Project

This project has the 3 parts of the assignment:

- baseline
- fine-tuning
- RAG

## What you need before running it

To run the notebooks, you need:

- Google Colab
- access to a GPU in Colab
- a Hugging Face account
- a Hugging Face token
- Google Drive

If the notebook uses a Llama model from Hugging Face, you may also need model access approval first.

## Files needed in Google Drive

Before running the notebooks, I uploaded these files to Google Drive:

- `dataset_clean.csv`
- `Austrian Tax Law LLM Prompt Assignment.xlsx`
- `legal_docs/` folder with the legal PDF files for the RAG part

## Important

The notebooks expect Google Drive to be mounted in Colab.
Some of them also require logging into Hugging Face with a token.

So before running the code:

1. open the notebook in Colab
2. mount Google Drive
3. log into Hugging Face if needed
4. make sure the file paths in the notebook match the files in Drive

## Folder structure

- `baseline/` baseline notebook
- `fine_tune/` fine-tuning notebook
- `rag/` RAG notebook and legal documents
- `results/` output csv files

## Results

The `results/` folder contains the generated output files and comparison files.
