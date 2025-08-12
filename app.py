# app.py (verze pro Render)
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

# Cesta k databázi. Render vytvoří dočasné úložiště, kam se databáze nahraje.
CHROMA_DB_PATH = "./chroma_db"
# ZMĚŇTE PODLE VAŠÍ SKUTEČNÉ KOLEKCE
COLLECTION_NAME = "knowledge_base" # <-- DŮLEŽITÉ: Změňte na název vaší kolekce!

# --- INICIALIZACE APLIKACE ---

app = Flask(__name__)
CORS(app)

model = genai.GenerativeModel('gemini-1.5-flash')
collection = None

# Pokus o připojení k databázi
try:
    # Kontrola, zda složka existuje
    if os.path.exists(CHROMA_DB_PATH):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Úspěšně připojeno ke kolekci '{COLLECTION_NAME}' s {collection.count()} dokumenty.")
    else:
        print(f"Chyba: Složka '{CHROMA_DB_PATH}' nebyla nalezena.")

except Exception as e:
    print(f"Chyba při připojování k ChromaDB: {e}")


# --- DEFINICE API ENDPOINTU ---

@app.route('/')
def index():
    # Jednoduchý endpoint pro ověření, že server běží
    return "Backend server pro chatbot je online!"


@app.route('/chat', methods=['POST'])
def chat():
    if not collection:
        return jsonify({"error": "ChromaDB kolekce není k dispozici. Zkontrolujte logy serveru."}), 500

    data = request.get_json()
    user_query = data.get('message')

    if not user_query:
        return jsonify({"error": "Chybí zpráva v požadavku."}), 400

    print(f"Přijat dotaz: {user_query}")

    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=5
        )
        context_documents = results['documents'][0]
        context = "\n\n".join(context_documents)
    except Exception as e:
        print(f"Chyba při dotazování do ChromaDB: {e}")
        return jsonify({"error": "Nepodařilo se prohledat databázi znalostí."}), 500

    prompt = f"""
    Jsi AI asistent. Odpověz na otázku uživatele pouze a výhradně na základě poskytnutého kontextu.
    Pokud odpověď v kontextu nenajdeš, upřímně napiš: "Omlouvám se, ale na tuto otázku neznám odpověď na základě dostupných informací."
    Buď stručný a věcný.

    --- KONTEXT ---
    {context}
    --- KONEC KONTEXTU ---

    OTÁZKA UŽIVATELE: {user_query}
    """

    try:
        response = model.generate_content(prompt)
        bot_response = response.text
    except Exception as e:
        print(f"Chyba při volání Gemini API: {e}")
        return jsonify({"error": "Chyba při komunikaci s AI modelem."}), 500

    print(f"Odpověď od Gemini: {bot_response}")

    return jsonify({'reply': bot_response})

# Poznámka: Blok if __name__ == '__main__': zde již není potřeba,
# protože o spuštění se stará Gunicorn na serveru Render.
