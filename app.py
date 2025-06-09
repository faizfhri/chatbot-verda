from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

app = Flask(__name__)
CORS(app)

# === 1. Inisialisasi Embedding Model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# === 2. Konfigurasi Supabase dan Chutes ===
SUPABASE_FAQ_URL = "https://gncwiljqegllybkibmeq.supabase.co/functions/v1/rag-chatbot"
CHUTES_API_URL = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")

# === 3. Fungsi untuk memanggil model via Chutes ===
def call_chutes_model(prompt):
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHUTES_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Gagal dari Chutes API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error saat memanggil Chutes API: {str(e)}"

# === 4. Ambil dan cari data dari Supabase ===
def retrieve_answer(query, top_k=3):
    try:
        query_embedding = embedding_model.encode(query).tolist()

        response = requests.post(
            SUPABASE_FAQ_URL,
            json={"query_embedding": query_embedding, "match_count": top_k}
        )

        if response.status_code != 200:
            return f"Gagal mengambil data dari Supabase: {response.status_code}"

        faqs = response.json()
        if not faqs:
            return "Tidak ada data relevan yang ditemukan."

        context_list = [
            f"- {faq.get('answer', '')} (Referensi: {faq.get('reference', 'tidak tersedia')})"
            for faq in faqs
        ]
        return "\n".join(context_list)

    except Exception as e:
        return f"Terjadi kesalahan saat mengambil data: {str(e)}"

# === 5. Format Prompt ===
def format_prompt(query, context):
    return (
        "Anda adalah chatbot edukasi berkelanjutan yang menjawab pertanyaan mahasiswa dengan informasi yang akurat.\n"
        "Gunakan hanya informasi dari konteks berikut dan jangan mengarang jawaban.\n"
        "Jika tidak ada informasi, jawab kamu tidak memiliki informasi tersebut dengan sopan.\n\n"
        f"Konteks:\n{context}\n\n"
        f"Pertanyaan:\n{query}\n\n"
        "Jawaban (gunakan format yang rapi dan informatif, tanpa menggunakan markdown seperti **bold**, _italic_, atau format khusus lainnya dan kalau ada [cite: angka random] itu hilangkan, (sebutkan referensinya dalam bentuk link, jika tidak tersedia jangan tampilkan)):\n"
    )

# === 6. State sederhana untuk menyimpan 1 history (opsional) ===
chat_history = []

# === 7. Endpoint untuk Chatbot ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Pesan tidak boleh kosong."})

    # Ambil jawaban dari Supabase (top-k FAQ)
    context = retrieve_answer(user_message, top_k=3)

    # Tambahkan ke history lokal
    chat_history.append({"query": user_message, "response": context})
    if len(chat_history) > 2:
        chat_history.pop(0)

    # Gabungkan semua respons sebelumnya (jika mau pakai konteks historis)
    combined_context = "\n".join([item["response"] for item in chat_history])
    prompt = format_prompt(user_message, combined_context)

    # Panggil LLM
    response_text = call_chutes_model(prompt)

    # Simpan respons aktual ke history
    chat_history[-1]["response"] = response_text

    return jsonify({"response": response_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
