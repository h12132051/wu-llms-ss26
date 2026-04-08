import csv   # Modul zum Einlesen und Schreiben von .csv Dateien (wie Excel)
import os    # Erlaubt dem Programm, mit dem Betriebssystem zu sprechen (z. B. Ordner erstellen)
import torch # PyTorch Bibliothek: Das mathematische Herzstück für Künstliche Intelligenz
from transformers import pipeline # Lädt das 'pipeline' Werkzeug, um blitzschnell fertige KIs herunterzuladen

def main():
    """
    Hauptfunktion für die Durchführung der Inferenz (Beantwortung von Fragen) ohne Kontext (Zero-Shot).
    Liest Fragen aus einer CSV-Datei ein, generiert Antworten per LLM und speichert die Ergebnisse.
    """
    
    # 1. HARDWARE BESCHLEUNIGUNG (DEVICE) WÄHLEN
    device = "cpu" # Standard: Der normale Hauptprozessor (sehr langsam für KI)
    if torch.backends.mps.is_available():
        device = "mps" # Falls es ein Apple Mac mit M-Chip ist, nutzen wir den extrem schnellen Grafikchip (MPS)
    elif torch.cuda.is_available():
        device = "cuda" # Falls ein Nvidia-Chip verbaut ist, nutzen wir CUDA (Grafikkartenbeschleuniger)

    # 2. MODELL HERUNTERLADEN UND STARTEN
    print(f"Loading model: google/flan-t5-base on {device}")
    
    # 'text2text-generation': Sagt der Pipeline, dass wir Text-In und Text-Out wollen
    # 'model="google/flan-t5-base"': Zieht ein etwas schlaueres Basis-Modell ("base" statt "small") aus dem Internet
    # 'device=device': Legt fest, dass dieses Modell auf dem gewählten Chip (GPU/CPU) laufen soll
    # 'framework="pt"': Wir sagen ihm "pt", was bedeutet es soll das PyTorch-Framework benutzen
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=device, framework="pt")
    
    # 3. PFADE DEFIENIEREN
    dataset_path = "../../data/dataset_clean.csv" # Hier liegt unser Test-Dokument mit den 644 Fragen
    output_path = "../results/model_1_output.csv" # Hier landen die KI-Antworten am Ende
    
    # Fallback, falls die Datei nicht am exakten Pfad liegt, sondern direkt im Skript-Ordner
    if not os.path.exists(dataset_path):
        dataset_path = "dataset_clean.csv" 
        
    print(f"Reading dataset from {dataset_path}")
    
    # 4. AUSGABE-DATEI (CSV) VORBEREITEN
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Erstellt den 'results' Ordner, falls er noch nicht da ist
    
    # Wir öffnen sofort die Ausgabe-Datei, um die erste Zeile, die "Spaltenüberschriften", zu schreiben
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"]) # Spalten heißen 'id' und 'answer'
        writer.writeheader() # Schreibt tatsächlich "id,answer" als erste Zeile in die Datei

    results_count = 0 # Zählt mit, wie viele Fragen wir schon beantwortet haben
    
    # 5. DER INFERENCE LOOP (FRAGEN BEANTWORTEN)
    # Wir öffnen den Datensatz ("r" = lesen) UND unsere Ausgabedatei ("a" = append/anhängen) gleichzeitig
    with open(dataset_path, "r", encoding="utf-8-sig") as f, open(output_path, "a", encoding="utf-8", newline="") as out_f:
        
        reader = csv.DictReader(f) # Macht aus dem Text eine liste von ansteuerbaren Spalten (Dictionary)
        writer = csv.DictWriter(out_f, fieldnames=["id", "answer"]) # Macht die Ausgabe-Datei bereit
        
        # Schleife (Loop): Wir gehen nun exakt über jede der 644 Fragen aus der Liste
        for row in reader:
            q_id = row['id']                         # Holt sich aus der jetzigen Zeile die Fragen-ID (z.B. CORP-TAX-001)
            question = row.get('prompt', '')         # Holt sich die eigentliche Frage ('prompt') in der CSV
            
            # 6. DEN PROMPT (DIE DIREKTIVE AN DIE KI) ERSTELLEN
            # Wir zwingen die nackte KI dazu, zumindest so zu tun, als wüsste sie, dass es um Steuern geht
            prompt = f"Answer the following question about Austrian tax: {question}"
            
            # 7. GENERIERUNG (INFERENCE)
            # Wir rufen 'generator()' auf. Das ist unser geladenes KI-Gehirn. 
            # 'max_length=150' bedeutet: Die Antwort darf nicht länger als 150 Wörter (Tokens) sein.
            out = generator(prompt, max_length=150)
            
            # Die Pipeline gibt eine Liste zurück, wir ziehen uns den ersten (und einzigen) Treffer heraus
            answer = out[0]['generated_text']
            
            # 8. ANTWORT ITERATIV IN DER DATEI SPEICHERN
            # iterativ = Für den Fall, dass der PC abstürzt, ist alles bis hierher sicher auf der Festplatte
            writer.writerow({'id': q_id, 'answer': answer}) # Schreibt ID und fertige Antwort rein
            out_f.flush()                                   # Zwingt die Festplatte, die Datei sofort zu speichern
            
            results_count += 1
            if results_count % 50 == 0:
                print(f"Processed {results_count} questions...") # Gibt alle 50 Runden einen kurzen Zwischenstand aus
                
    print(f"Finished generating {results_count} answers.")
    print(f"Saved to {output_path}")

# Startpunkt: Wenn Python dieses Skript startet, soll es exakt die Funktion main() oben aufrufen
if __name__ == "__main__":
    main()
