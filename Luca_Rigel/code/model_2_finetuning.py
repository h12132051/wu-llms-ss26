import pandas as pd                             # Für das Arbeiten mit Tabellen (CSV, Excel)
import torch                                    # Das PyTorch Framework (Herzstück für KI-Berechnungen)
from torch.utils.data import Dataset            # Wichtiges PyTorch-Modul zum Strukturieren von Trainingsdaten
from sklearn.model_selection import train_test_split # Hilft dabei, Daten in Train/Test Sets zu zerschneiden

from transformers import (                      # Die "Hugging Face" Bibliothek für fertige Sprachmodelle
    AutoTokenizer,                              # Übersetzt Wörter in Zahlen
    AutoModelForCausalLM,                       # Lädt ein Sprachmodell (hier GPT-basierend), das Text vorhersagt
    Trainer,                                    # Die fertige Trainings-Schleife von Hugging Face
    TrainingArguments,                          # Alle Einstellungen für das Training (Epochen, Lernrate etc.)
    DataCollatorForLanguageModeling             # Formatiert Input-Daten effizient für das Training
)

import os                                       # Für Dateipfade

# -----------------------------
# 1. Load and split dataset / Daten laden & aufteilen
# -----------------------------
# -----------------------------
# 1. PDF Daten einlesen & In Texte umwandeln
# -----------------------------
# Wir laden die PDF-Lehrbücher und nutzen den Text direkt als Trainings-Futter!
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("Bitte führe '!pip install pypdf' in Colab aus, damit die PDFs gelesen werden können!")

def get_pdf_text(data_dir):
    texts = []
    if HAS_PYPDF:
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                print(f"Lese PDF für Training: {file}")
                reader = PdfReader(os.path.join(data_dir, file))
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        # Wir splitten in Absätze, damit das Modell mundgerechte Stücke bekommt
                        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
                        texts.extend(paragraphs)
    return texts

data_dir = "../../data"
if not os.path.exists(data_dir):
    data_dir = "data"

all_texts = get_pdf_text(data_dir)

if not all_texts:
    print("Warnung: Keine PDF-Texte gefunden! Das Modell wird nur auf dem Basis-Wissen antworten.")
    all_texts = ["Österreichisches Steuerrecht ist komplex.", "Die kalte Progression wurde abgeschafft."]

# Aufteilen in Train und Val (Zufällig)
train_texts, val_texts = train_test_split(all_texts, test_size=0.1, random_state=42)

# -----------------------------
# 2. Custom Dataset (Vorbereiten der Daten für PyTorch)
# -----------------------------
class LegalTextDataset(Dataset):
    """
    Eigene Dataset-Klasse für PyTorch, um die PDF-Texte in Zahlen (Tensoren) umzuwandeln.
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Das Modell lernt einfach den Text der Lehrbücher auswendig/zu verstehen
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        input_ids = encoding["input_ids"]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(input_ids),
        }

# -----------------------------
# 3. Tokenizer & Model / Lade Architektur
# -----------------------------
model_name = "distilgpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------------
# 4. Datasets initialisieren
# -----------------------------
train_dataset = LegalTextDataset(train_texts, tokenizer)
val_dataset = LegalTextDataset(val_texts, tokenizer)

# -----------------------------
# 5. Training Arguments / Konfiguration für das Fine-Tuning
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,                         # Bei PDFs reicht oft 1 Epoche, um das Vokabular zu lernen
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",                         # Wir sparen Zeit in Colab
    logging_steps=50,
    save_strategy="no",                         # Wir speichern keine riesigen Zwischenschritte (spart Platz)
    warmup_steps=10,
    report_to="none",
)

# -----------------------------
# 6. Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# 7. Trainer Instanziieren
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# -----------------------------
# 8. Train (Training)
# -----------------------------
if all_texts:
    print(f"Starte Training auf {len(all_texts)} Text-Abschnitten aus deinen PDFs...")
    trainer.train()

# -----------------------------
# 9. Test inference / Inferencing Loop für dataset_clean.csv
# -----------------------------
def generate_predictions():
    """
    Diese Funktion nutzt das GERADE EBEN fertig trainierte Modell, lädt unsere echten 644 Steuer-Aufgaben 
    (die Uni-Questions) und speichert die finalen Antworten ab.
    """
    dataset_path = "../../data/dataset_clean.csv"
    output_path = "../results/model_2_output.csv"
    
    if not os.path.exists(dataset_path):
        dataset_path = "dataset_clean.csv"
        
    print(f"Lese Datensatz von {dataset_path} für Vorhersagen...")
    df_test = pd.read_csv(dataset_path)         # Lädt die offiziellen Uni-Fragen
    
    # Bereitet die leere CSV Ausgabedatei vor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(columns=["id", "answer"]).to_csv(output_path, index=False)
    
    model.eval()                                # WICHTIG: Sagt dem Modell "Nicht mehr lernen! Ab jetzt nur noch abrufen (Inferenz)!"
    
    # Packt das geladene Modell auf die beste vorhandene Hardware (Grafikkarte)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
        
    device = next(model.parameters()).device    # Findet heraus wo das Modell jetzt tatsächlich liegt
    
    results_count = 0
    # iterrows() iteriert (durchläuft) Zeile für Zeile unsere Test-Fragen.
    for i, row in df_test.iterrows():
        q_id = row['id']
        question = row.get('prompt', '')
        
        # Das exakte Format in dem die Frage an die KI geschickt wird
        prompt = f"Frage: {question}\nAntwort:"
        
        # Wandelt den Prompt in Tokens/Zahlen um
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Schiebt die Token-Werte direkt auf die schnelle Grafikkarte rüber
        
        # no_grad() spart sehr viel Arbeitsspeicher, da für Inference keine "Gradienten" (Lern-Kurven) berechnet werden müssen
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=150,             # Bitte antworte in max 150 Wörtern
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2          # Ein genialer Kniff: Verhindert, dass das kleine (nur gpt-2) Modell einen Satz immer und immer wiederholt
            )
        
        # Die Antwort der KI beginnt hinter unserem Input, also schneiden wir unseren eigenen Input von der Startklammer ab
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # 'decode' übersetzt die Token-Zahlen der KI (z.B. [144, 55, 99]) wieder zurück in echtes deutsches Text-Wort.
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Schreibt das direkt Zeile für Zeile sicher in die Output-CSV ("mode='a'" = Append, er hängt es immer unten an!)
        pd.DataFrame([{'id': q_id, 'answer': answer}]).to_csv(output_path, mode='a', header=False, index=False)
        results_count += 1
        
        if results_count % 50 == 0:
            print(f"  {results_count} Fragen beantwortet...")

    print(f"Fertig! {results_count} Vorhersagen in {output_path} gespeichert.")

# Ausführung starten
generate_predictions()
