from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import threading
import time
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)
CORS(app)

# === Embedding dan Memory ===
embedding_model = embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
memory = ConversationBufferMemory(k=3)

# === Konfigurasi Supabase dan Chutes ===
SUPABASE_FAQ_URL = "https://gncwiljqegllybkibmeq.supabase.co/functions/v1/rag-chatbot"
CHUTES_API_URL = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")

# === Model Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "Anda adalah chatbot edukasi berkelanjutan yang menjawab pertanyaan mahasiswa dengan informasi yang akurat.\n"
        "Gunakan hanya informasi dari konteks berikut dan jangan mengarang jawaban.\n"
        "Jika tidak ada informasi, jawab kamu tidak memiliki informasi tersebut dengan sopan (tidak usah minta detail tambahan atau lainnya, cukup bilang tidak memiliki informasi saja dan bilang apa informasi yang kamu punya dari konteks).\n\n"
        "Konteks:\n{context}\n\n"
        "Pertanyaan:\n{query}\n\n"
        "Jawaban (gunakan format yang rapi dan informatif, tanpa menggunakan markdown seperti **bold**, _italic_, atau format khusus lainnya):\n"
    )
)

# === Fungsi: Panggil LLM dari Chutes ===
def call_chutes_model(prompt):
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHUTES_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Gagal dari Chutes API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error saat memanggil Chutes API: {str(e)}"

# === Fungsi: Ambil FAQ dari Supabase ===
def retrieve_answer(query, top_k=3):
    try:
        response = requests.get(SUPABASE_FAQ_URL)
        if response.status_code != 200:
            return "Gagal mengambil data dari Supabase."
        faqs = response.json()
        if not faqs:
            return "Tidak ada data FAQ di Supabase."

        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        faq_texts = [f"{faq['question']} {faq['answer']}" for faq in faqs]
        faq_embeddings = embedding_model.encode(faq_texts, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
        top_results = scores.topk(k=top_k)

        context_list = [faqs[int(idx)].get("answer", "") for idx in top_results.indices]
        return "\n".join(context_list) if context_list else "Tidak ada informasi yang relevan."
    except Exception as e:
        return f"Terjadi kesalahan saat mengambil data: {str(e)}"

# === Endpoint Flask ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"response": "Pesan tidak boleh kosong."})

    context = retrieve_answer(user_message, top_k=3)
    memory.save_context({"query": user_message}, {"response": context})
    history = memory.load_memory_variables({}).get("history", "")

    formatted_prompt = prompt_template.format(query=user_message, context=f"{history}\n{context}")
    response_text = call_chutes_model(formatted_prompt)

    memory.save_context({"query": user_message}, {"response": response_text})
    return jsonify({"response": response_text})

# === Background Thread untuk Reset Memory ===
def clear_memory_periodically():
    while True:
        time.sleep(1800)  # 1800 detik = 30 menit
        memory.clear()
        print("🧹 Memory dibersihkan otomatis (30 menit)")

threading.Thread(target=clear_memory_periodically, daemon=True).start()

# === Jalankan Flask App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
