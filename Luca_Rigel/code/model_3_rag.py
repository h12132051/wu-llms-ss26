import pandas as pd                             # Arbeitet mit Tabellen
from transformers import pipeline               # Zum super simplen Laden von Language Models (LLMs)
import torch                                    # Das tiefe Grund-Framework für tensor / vector Matrizen
import os                                       # Ordner und Dateipfade anlegen

try:
    from sklearn.feature_extraction.text import TfidfVectorizer # Wandelt alle Wörter in Vektoren/Kalkulationen um (Wie nah ist Text A an Text B verwandt)
    from sklearn.metrics.pairwise import cosine_similarity      # Berechnet die Winkel/Nähe zwischen zwei Text-Vektoren (Sucht den Kontext)
    from tqdm import tqdm                                       # Produziert extrem coole Fortschrittsbalken im Terminal
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False                                         # Wenn scikit-learn nicht installiert wurde vom User, failen wir trotzdem nicht ab

try:
    from pypdf import PdfReader                        # Liest Text aus PDF-Dateien (Wichtig für die neuen Quellen des Users!)
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

def simple_retrieve(query, docs, top_k=1):
    """
    Einfache regelbasierte Suchfunktion (Fallback, falls sklearn nicht verfügbar ist).
    Sucht nach den meisten gleichen Wörtern zwischen der Frage und den Texten (Wort-Überschneidung).
    """
    query_words = set(query.lower().split())    # Schneidet die Frage in ganz kleine Wörter und ignoriert exakte Groß/Kleinschreibung
    scores = []
    
    # Loop über ganzes Gesetzes-Dokument
    for doc in docs:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words)) # Schaut wie viele der Frage-Wörter auch exakt ident im Gesetzestext stehen
        scores.append((overlap, doc))           # Speichert Score (1 Punkt pro Gleichem Wort)
        
    scores.sort(key=lambda x: x[0], reverse=True)              # Sortiert die Treffer-Liste absteigend 
    return [doc for score, doc in scores[:top_k] if score > 0] # Liefert nur den "top_k" besten Treffer zurück (idR den ersten Platz!)

def load_knowledge_base(data_dir):
    """
    Lädt Lehrbücher (PDF) aus dem data-Ordner
    und baut eine Liste (Corpus) von Texten auf.
    RAG = Retrieval (Suche) - Augmented (Angereichertes) - Generation (Generieren von Antworten).
    """
    corpus = [] # Array für alle gefundenen Gesetzestexte
    
    # 2. PDF DATEN LADEN (Die neuen Quellen!)
    if HAS_PYPDF:
        print("Searching for PDF sources in data directory...")
        for file in sorted(os.listdir(data_dir)):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file)
                print(f"Reading PDF: {file}...")
                try:
                    reader = PdfReader(pdf_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text and len(text) > 20:
                            # Wir splitten die PDF in Absätze, damit die Suche präziser wird
                            paragraphs = text.split("\n\n")
                            corpus.extend([p.strip() for p in paragraphs if len(p) > 20])
                except Exception as e:
                    print(f"Could not read PDF {pdf_path}: {e}")
    else:
        print("Note: pypdf not installed. Please run '!pip install pypdf' in Colab to use PDF sources.")

    # "Set" filtert absolut alle Text-Doppelungen im Memory raus.
    corpus = list(set([doc for doc in corpus if len(doc) > 10]))        
    return corpus
def main():
    """
    Hauptfunktion für den RAG-Prozess.
    1. Läd das LLM-Generierungsmodell (T5 small)
    2. Läd unsere Textsammlung (Lexikon)
    3. Führt die eigentlichen Tests durch und "befragt sich vorher beim Lexikon".
    """
    
    # -- HARDWARE AUSWAHL (Grafikbeschleunigung) --
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"  # Mac GPU (Extreme Zeitersparrnis bei RAG!)
    elif torch.cuda.is_available():
        device = "cuda" # Nvidia Karte

    print(f"Loading Hugging Face pipeline on {device}...")
    # Generator Initialisierung: das Herzstück unseres "dummen" Generatoren (welcher nur formuliert)
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=device, framework="pt")

    data_dir = "../../data"
    print("Loading Knowledge Base...")
    corpus = load_knowledge_base(data_dir) # 3000 bis 50.000 Sätze werden hier in Arrays (den Ram Speicher!) geladen.
    print(f"Loaded {len(corpus)} unique documents into knowledge base.")
    
    if not corpus:
        # Mini-Check ob alles geklappt hat, ansonsten würden wir einfach weiterlaufen und Müll ausspucken
        print("Warning: Knowledge base is empty. Proceeding without context.")
        corpus = ["No information available."]
    
    # -- VECTORIZER UND SUCH-SETUP (Retrieval) --
    vectorizer = None
    tfidf_matrix = None
    if HAS_SKLEARN:
        # TF-IDF (Term Frequency-Inverse Document Frequency)
        # Baut eine mathematische Landschaft auf. Ein extrem seltenes Steuer-Begriff bringt hier 
        # mehr Punkte als Worte wie 'eine', 'oder', 'und' welche man rausfiltert ('stop_words=english', in dt. leider mäßig)
        print("Building TF-IDF Vectorizer for retrieval...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(corpus)  # Verarbeitet einmal am Start in einer dicken Rechnung sofort ALLE texte.
    else:
        print("sklearn not found. Using simple keyword overlap for retrieval.")

    test_data_path = "../../data/dataset_clean.csv"
    output_path = "../results/model_3_output.csv"
    
    print(f"Loading test dataset from {test_data_path}")
    if os.path.exists(test_data_path):
        df_test = pd.read_csv(test_data_path, encoding='utf-8-sig') # Lade Uni Test
    else:
        print(f"Error: Could not find {test_data_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(columns=["id", "answer"]).to_csv(output_path, index=False) # Erzeugen der Datei mit Spaltenkopf

    print("Running RAG Inference...")
    
    iterable = df_test.iterrows()
    if HAS_SKLEARN:
        iterable = tqdm(iterable, total=len(df_test)) # tqdm = Ein Terminal-Ladebalken für Iteratoren <3!
        
    results_count = 0
    # Schleife durch jede Uni-Prüfungsfrage!
    for index, row in iterable:
        q_id = row['id']
        question = row.get('prompt', row.get('text', row.get('question', '')))

        if pd.isna(question) or str(question).strip() == "":
            answer = ""
        else:
            # ---> RAG SCHRITT 1 / RETRIEVAL: SUCHE KONTEXT <-------
            if HAS_SKLEARN:
                # 1.1 Mache aus dem Fragensatz von der Uni einen Vektor.
                question_vec = vectorizer.transform([str(question)])
                
                # 1.2 Multipliziere und vergleiche Winkel (=Cosine Similarity) der Frage mit allen Datensätzen
                similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
                
                # 1.3 Welcher Index (welches Text Dokument in unserem Array) hatte den am weitesten übereinstimmenden Winkel? Den nehmen wir via argmax!
                best_doc_idx = similarities.argmax()
                
                # 1.4 Lade diesen spezifischen Text in eine String Variable !
                retrieved_context = str(corpus[best_doc_idx])
            else:
                docs = simple_retrieve(str(question), corpus, top_k=1)
                retrieved_context = docs[0] if docs else ""
            
            # ---> RAG SCHRITT 2 / AUGMENTATION: BAUE EINEN CLEVEREN PROMPT <-------
            # Verhindern das der Text den Prompt sprengt (OOM = Out of memory errors bei Grafikkarten!)
            if len(retrieved_context) > 1000:
                retrieved_context = retrieved_context[:1000] + "..."

            # Wir füttern das "t5-small" ZeroShot modell mit Text der es absolut allwissend macht.
            prompt = f"Context: {retrieved_context}\\n\\nQuestion: {question}\\nAnswer:"
            
            # ---> RAG SCHRITT 3 / GENERATION: LASS IHN SCHREIBEN <-------
            try:
                # LLM generiert Antwort strikt nach dem im Kontext stehenden wissen.
                out = generator(prompt, max_length=50, num_return_sequences=1)
                answer = out[0]['generated_text']
            except Exception as e:
                print(f"Error generating answer for ID {q_id}: {e}")
                answer = "Error generating answer"

        # Sicher und iterativ hinten dran speichern!
        pd.DataFrame([{'id': q_id, 'answer': answer}]).to_csv(output_path, mode='a', header=False, index=False)
        results_count += 1

    print(f"Successfully saved {results_count} rows to {output_path}")

# Nur wenn die RAG Py in Python als Main ausgeführt wird (und nicht als import) -> Startschuss
if __name__ == "__main__":
    main()
