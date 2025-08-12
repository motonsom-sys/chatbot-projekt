# app.py (opravená verze pro Render)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
import google.generativeai as genai

# --- KONFIGURACE ---

# Načtení API klíče z "Environment Variable" na Renderu
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Tato chyba se zobrazí v logu na Renderu, pokud klíč chybí
    raise ValueError("Není nastaven GOOGLE_API_KEY v prostředí Renderu.")

genai.configure(api_key=api_key)

# Cesta k databázi. Musí odpovídat cestě vytvořené skriptem create_chromadb.py
CHROMA_DB_PATH = "./chroma_db"
# Název kolekce, jak je definován ve vašem skriptu pro vytvoření databáze
COLLECTION_NAME = "vylety_kolekce" 

# --- INICIALIZACE APLIKACE ---
app = Flask(__name__)
CORS(app)

model = genai.GenerativeModel('gemini-1.5-flash')
collection = None

# Pokus o připojení k databázi po spuštění serveru
try:
    # Kontrola, zda složka s databází existuje
    if os.path.exists(CHROMA_DB_PATH):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Úspěšně připojeno ke kolekci '{COLLECTION_NAME}' s {collection.count()} dokumenty.")
    else:
        print(f"Chyba: Složka s ChromaDB databází '{CHROMA_DB_PATH}' nebyla nalezena. Ujistěte se, že 'Build Command' na Renderu spustí 'create_chromadb.py'.")

except Exception as e:
    print(f"Chyba při připojování k ChromaDB: {e}")
    # Nastavení kolekce na None, aby se později vyvolala chyba, pokud je to potřeba
    collection = None


# --- DEFINICE API ENDPOINTU ---

@app.route('/')
def index():
    # Jednoduchý endpoint pro ověření, že server běží
    return "Backend server pro chatbot je online!"


@app.route('/chat', methods=['POST'])
def chat():
    # Zkontrolujeme, zda je kolekce dostupná
    if not collection:
        return jsonify({"error": "Chyba: ChromaDB kolekce není k dispozici. Zkontrolujte logy serveru."}), 500

    # Získání uživatelského dotazu z JSON těla požadavku, s klíčem 'query'
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Chybí zpráva v požadavku. Očekává se klíč 'query'."}), 400

    print(f"Přijat dotaz: {user_query}")

    try:
        # 1. Vyhledání relevantního kontextu v ChromaDB
        results = collection.query(
            query_texts=[user_query],
            n_results=3
        )
        context_documents = results['documents'][0]
        context = "\n\n---\n\n".join(context_documents)
        
        # 2. Vytvoření promptu pro Gemini API
        prompt = f"""
            Jsi Profesor Nsom, laskavý a nápomocný kronikář, který odpovídá na dotazy o cestách motorkářského klubu MOTONSOM.
            Odpovídej pouze na základě poskytnutých informací v sekci 'Kontext'. Pokud kontext neobsahuje odpověď, přiznej to a nehádej.
            Odpovědi by měly být krátké, maximálně 3-4 věty.
            V odpovědi na dotazy o lidech uváděj celé jméno a nick z motonsomu, pokud je k dispozici.
            V kontextu se nachází informace o tom, co se stalo, kdy a s kým. Soustřeď se na tyto informace.
            Vždy odpovídej v češtině.

            Kontext:
            {context}

            Dotaz:
            {user_query}
        """

        # 3. Volání Gemini API
        response = model.generate_content(prompt)
        bot_response = response.text
        
    except Exception as e:
        print(f"Chyba při volání Gemini API nebo dotazování ChromaDB: {e}")
        return jsonify({"error": "Došlo k chybě při generování odpovědi."}), 500

    print(f"Odpověď od Gemini: {bot_response}")

    # Vrácení odpovědi s klíčem 'response', aby odpovídal frontendu
    return jsonify({'response': bot_response})
