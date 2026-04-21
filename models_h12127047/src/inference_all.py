import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. SETUP
ft_model_path = "./fine_tuned_legal" # Dein trainiertes Modell
base_model_name = "dbmdz/german-gpt2" # Das ursprüngliche Basis-Modell
test_data_url = "https://raw.githubusercontent.com/svakulenk0/wu-llms-ss26/main/dataset_clean.csv"
knowledge_path = "training_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TOKENIZER LADEN ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- MODELLE LADEN ---
print("Lade Basis-Modell...")
model_base = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

print("Lade Fine-Tuned Modell...")
model_ft = AutoModelForCausalLM.from_pretrained(ft_model_path)
model_ft.resize_token_embeddings(len(tokenizer)) # Fix für Padding-Tokens
model_ft.to(device)

model_base.eval()
model_ft.eval()

# Daten laden
df_test = pd.read_csv(test_data_url)
df_knowledge = pd.read_csv(knowledge_path, sep=';').dropna(subset=['train'])

# 2. RAG LOGIK
def get_context(question):
    question_lower = str(question).lower()
    for _, row in df_knowledge.iterrows():
        ref = str(row['Full Reference']).lower()
        if ref in question_lower or any(word in question_lower for word in ref.split() if len(word) > 3):
            return str(row['train'])[:700]
    return ""

# 3. GENERIERUNG (mit Token-Anpassung)
def generate_answer(model, question, mode="ft"):
    # Unterschiedliche Max-Tokens wie von dir angemerkt
    tokens_to_generate = 200 if mode == "inference_only" else 100
    
    if mode == "rag":
        context = get_context(question)
        prompt = f"Hintergrund: {context}\nFrage: {question}\nAntwort:"
    else:
        prompt = f"Frage: {question}\nAntwort:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=tokens_to_generate, 
            temperature=0.2, 
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("Antwort:")[-1].strip() if "Antwort:" in full_text else full_text.strip()

# 4. LOOP (Alle 3 Varianten)
results_inf = []
results_ft = []
results_rag = []

print(f"Starte Generierung für {len(df_test)} Fragen...")

for idx, row in df_test.iterrows():
    q_id = row['id']
    q_text = row['question']
    
    # 1. Base Model Only (Inference Only) - Mehr Tokens!
    ans_inf = generate_answer(model_base, q_text, mode="inference_only")
    results_inf.append({"id": q_id, "model_answer": ans_inf})
    
    # 2. Fine-Tune Only
    ans_ft = generate_answer(model_ft, q_text, mode="ft")
    results_ft.append({"id": q_id, "answer": ans_ft})
    
    # 3. RAG (Fine-Tune + Kontext)
    ans_rag = generate_answer(model_ft, q_text, mode="rag")
    results_rag.append({"id": q_id, "answer": ans_rag})
    
    if idx % 50 == 0:
        print(f"Fortschritt: {idx}/{len(df_test)}")

# 5. EXPORT
pd.DataFrame(results_inf).to_csv("inference_only.csv", index=False)
pd.DataFrame(results_ft).to_csv("submission_fine_tuned.csv", index=False)
pd.DataFrame(results_rag).to_csv("submission_rag.csv", index=False)

print("✅ Alle drei Dateien (Inference Only, FT, RAG) wurden erstellt.")