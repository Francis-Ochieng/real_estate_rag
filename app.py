# app.py

import streamlit as st
from stqdm import stqdm
import os, time, shutil
from ingestion import load_pdf, load_docx, load_txt, load_csv, load_url, load_youtube_transcript
from utils import simple_chunk_text
from retriever import FAISSRetriever, FAISS_DIR
from dotenv import load_dotenv
from groq import Groq

# Use the updated LangChain imports
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------- Setup ----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

st.set_page_config(page_title="Real Estate RAG Assistant", layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []


# ---------------- Sidebar Settings ----------------
st.sidebar.title("⚙️ Settings")
chunk_size = st.sidebar.number_input(
    "Chunk size (chars)", min_value=200, max_value=10000, value=1200, step=100
)
overlap = st.sidebar.number_input(
    "Chunk overlap (chars)", min_value=0, max_value=chunk_size // 2, value=200, step=50
)
top_k = st.sidebar.number_input("Top-K passages", min_value=1, max_value=20, value=5)
use_reranker = st.sidebar.checkbox(
    "Use reranker (Groq does not support yet)", value=False
)

groq_models = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
selected_model = st.sidebar.selectbox("Choose Groq model:", groq_models, index=0)


# ---------------- Main UI ----------------
st.title("🏠 Real Estate RAG Assistant")
st.markdown(
    "Upload documents, add URLs or YouTube links, then click **Ingest**. "
    "After ingestion, ask a question below."
)

uploads = st.file_uploader(
    "📂 Upload files (PDF, DOCX, TXT, CSV)", accept_multiple_files=True
)
urls_text = st.text_area("🌐 Add URLs or YouTube links (one per line)", height=120)
ingest_btn = st.button("🚀 Ingest documents")


def source_icon(src: str) -> str:
    src = src.lower()
    if src.endswith(".pdf"):
        return "📄"
    elif "youtube" in src or "youtu.be" in src:
        return "🎥"
    elif src.startswith("http"):
        return "🌐"
    return "📑"


# ---------------- Ingestion ----------------
if ingest_btn:
    with st.spinner("Ingesting..."):
        retriever = FAISSRetriever(use_reranker=use_reranker)
        retriever.reset_collection()

        all_chunks, metadatas = [], []

        # Handle uploaded files
        for f in uploads or []:
            name = f.name
            ext = name.split(".")[-1].lower()
            temp_path = "tmp_upload_" + name
            with open(temp_path, "wb") as out:
                out.write(f.getbuffer())

            if ext == "pdf":
                txt = load_pdf(temp_path)
            elif ext in ["docx", "doc"]:
                txt = load_docx(temp_path)
            elif ext == "txt":
                txt = load_txt(temp_path)
            elif ext == "csv":
                txt = load_csv(temp_path)
            else:
                txt = f"[unsupported file type: {ext}]"

            if txt and txt.strip() and not txt.startswith("[ERROR"):
                chunks = simple_chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
                for i, ch in enumerate(chunks):
                    all_chunks.append(ch)
                    metadatas.append({"source": f"📄 {name}", "chunk": i})
                st.info(f"✅ Ingested file: {name} ({len(chunks)} chunks)")
            else:
                st.warning(f"⚠️ No valid text extracted from: {name}")

            os.remove(temp_path)

        # Handle URLs / YouTube links
        for line in (urls_text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                if "youtube" in line or "youtu.be" in line:
                    txt = load_youtube_transcript(line)
                    src = f"🎥 YouTube: {line}"
                else:
                    txt = load_url(line)
                    src = f"🌐 Web: {line}"
                if txt and txt.strip() and not txt.startswith("[ERROR"):
                    chunks = simple_chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
                    for i, ch in enumerate(chunks):
                        all_chunks.append(ch)
                        metadatas.append({"source": src, "chunk": i})
                    st.info(f"✅ Ingested: {src} ({len(chunks)} chunks)")
                else:
                    st.warning(f"⚠️ No valid text extracted from: {line}")
            except Exception as e:
                st.error(f"❌ Failed to load {line}: {e}")

        # Store to FAISS
        if all_chunks:
            retriever.add_documents(all_chunks, metadatas=metadatas)
            retriever.persist()
            st.success(f"Ingested {len(all_chunks)} chunks into the FAISS index.")
            st.session_state["last_ingested"] = {
                "count": len(all_chunks),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            st.warning("⚠️ Nothing was ingested. Please check your files/links.")


# ---------------- Sidebar Info ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("📊 Ingested info:")
if "last_ingested" in st.session_state:
    st.sidebar.json(st.session_state["last_ingested"])


# ---------------- Chat + Query ----------------
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("💬 Ask a question about your documents:")
    if st.button("Send"):
        if not query:
            st.warning("Please type a question.")
        else:
            retriever = FAISSRetriever(use_reranker=use_reranker)
            items = retriever.query(query, top_k=top_k, rerank_top_n=top_k * 2)

            context = "\n\n".join(
                [
                    f"{source_icon(it['meta'].get('source',''))} "
                    f"Source: {it['meta'].get('source')}\n{it.get('text')[:1000]}"
                    for it in items
                ]
            )

            answer = None
            try:
                if client:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Use the given context to answer concisely.",
                        },
                        {
                            "role": "user",
                            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}",
                        },
                    ]
                    resp = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        max_tokens=300,
                        temperature=0.3,
                    )
                    answer = resp.choices[0].message.content.strip()
                else:
                    answer = "⚠️ No GROQ_API_KEY set — generation disabled."
            except Exception as e:
                answer = f"❌ Generation error: {e}"

            st.session_state["history"].append(
                {"query": query, "answer": answer, "retrieved": items}
            )

    if st.button("🧹 Clear chat"):
        st.session_state["history"] = []

    for turn in reversed(st.session_state["history"]):
        st.markdown(f"**Q:** {turn['query']}")
        st.info(turn["answer"])
        with st.expander("🔍 Show retrieved passages & scores"):
            for it in turn["retrieved"]:
                src = it["meta"].get("source", "unknown")
                st.write(f"{source_icon(src)} Source: {src} | distance: {it.get('distance'):.4f}")
                st.write(it["text"][:1000])


# ---------------- Documents Preview ----------------
with col2:
    st.markdown("### 📑 Documents preview")
    try:
        if os.path.exists(FAISS_DIR):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            shown = 0
            for doc in vs.docstore._dict.values():
                st.write(doc.metadata.get("source", "unknown"))
                st.write(doc.page_content[:300])
                shown += 1
                if shown >= 5:
                    break
            if shown == 0:
                st.write("Index exists but no documents found.")
        else:
            st.write("No FAISS index yet.")
    except Exception as e:
        st.error(f"❌ Preview failed: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear documents preview"):
        try:
            if os.path.exists(FAISS_DIR):
                shutil.rmtree(FAISS_DIR)
                st.success("✅ All documents cleared from FAISS index.")
                st.session_state["history"] = []
                st.rerun()
            else:
                st.info("No index to clear.")
        except Exception as e:
            st.error(f"Error clearing index: {e}")

    if st.button("⬇️ Download transcript of chat"):
        lines = []
        for turn in st.session_state["history"]:
            lines.append("Q: " + turn["query"])
            lines.append("A: " + (turn["answer"] or ""))
            lines.append("\n")
        txt = "\n".join(lines)
        st.download_button("Click to download", txt, file_name="chat_transcript.txt")
