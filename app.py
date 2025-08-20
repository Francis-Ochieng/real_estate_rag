import streamlit as st
from stqdm import stqdm
import os, time
from ingestion import load_pdf, load_docx, load_txt, load_csv, load_url, load_youtube_transcript
from utils import simple_chunk_text
from retriever import ChromaRetriever
import json

st.set_page_config(page_title='Real Estate RAG Assistant', layout='wide')

if 'history' not in st.session_state:
    st.session_state['history'] = []

st.sidebar.title('Settings')
chunk_size = st.sidebar.number_input('Chunk size (chars)', min_value=200, max_value=10000, value=1200, step=100)
overlap = st.sidebar.number_input('Chunk overlap (chars)', min_value=0, max_value=chunk_size//2, value=200, step=50)
top_k = st.sidebar.number_input('Top-K passages', min_value=1, max_value=20, value=5)
use_reranker = st.sidebar.checkbox('Use reranker (may download model)', value=True)

st.title('🏠 Real Estate RAG Assistant')
st.markdown('Upload documents, add URLs or YouTube links, then click **Ingest**. After ingestion ask a question below.')

uploads = st.file_uploader('Upload files (PDF, DOCX, TXT, CSV)', accept_multiple_files=True)
urls_text = st.text_area('Add URLs or YouTube links (one per line)', height=120)
ingest_btn = st.button('Ingest documents')

if ingest_btn:
    with st.spinner('Ingesting...'):
        retriever = ChromaRetriever(use_reranker=use_reranker)
        all_chunks = []
        metadatas = []
        for f in uploads or []:
            name = f.name
            ext = name.split('.')[-1].lower()
            temp_path = os.path.join('tmp_upload_'+name)
            with open(temp_path,'wb') as out:
                out.write(f.getbuffer())
            if ext == 'pdf':
                txt = load_pdf(temp_path)
            elif ext in ['docx','doc']:
                txt = load_docx(temp_path)
            elif ext in ['txt']:
                txt = load_txt(temp_path)
            elif ext in ['csv']:
                txt = load_csv(temp_path)
            else:
                txt = f"[unsupported file type: {ext}]"
            chunks = simple_chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
            for i,ch in enumerate(chunks):
                all_chunks.append(ch)
                metadatas.append({'source': name, 'chunk': i})
            os.remove(temp_path)
        # URLs and YouTube
        for line in (urls_text or '').splitlines():
            line = line.strip()
            if not line:
                continue
            if 'youtube' in line or 'youtu.be' in line:
                txt = load_youtube_transcript(line)
                src = line
            else:
                txt = load_url(line)
                src = line
            chunks = simple_chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
            for i,ch in enumerate(chunks):
                all_chunks.append(ch)
                metadatas.append({'source': src, 'chunk': i})
        # persist to vector DB
        retriever.add_documents(all_chunks, metadatas=metadatas)
        retriever.persist()
        st.success(f'Ingested {len(all_chunks)} chunks into the vector store.')
        st.session_state['last_ingested'] = {'count': len(all_chunks), 'time': time.time()}

st.sidebar.markdown('---')
st.sidebar.markdown('Ingested info:')
if 'last_ingested' in st.session_state:
    st.sidebar.write(st.session_state['last_ingested'])

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input('Ask a question about your documents:')
    if st.button('Send'):
        if not query:
            st.warning('Please type a question.')
        else:
            retriever = ChromaRetriever(use_reranker=use_reranker)
            items = retriever.query(query, top_k=top_k, rerank_top_n=top_k*2)
            # build context
            context = '\n\n'.join([f"Source: {it['meta'].get('source')}\n{it.get('text')[:1000]}" for it in items])
            # simple generation using OpenAI if available
            answer = None
            try:
                import openai
                if os.getenv('OPENAI_API_KEY'):
                    openai.api_key = os.getenv('OPENAI_API_KEY')
                    prompt = f"You are a helpful assistant. Use the following context to answer as concisely as possible.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer:"
                    resp = openai.Completion.create(model='text-davinci-003', prompt=prompt, max_tokens=300)
                    answer = resp.choices[0].text.strip()
                else:
                    answer = 'No OPENAI_API_KEY set — generation disabled. Retrieval result shown instead.'
            except Exception as e:
                answer = f'Generation error: {e}'

            # store in history
            st.session_state['history'].append({'query': query, 'answer': answer, 'retrieved': items})
    if st.button('Clear chat'):
        st.session_state['history'] = []

    # display history
    for i,turn in enumerate(reversed(st.session_state['history'])):
        st.markdown(f"**Q:** {turn['query']}")
        st.info(turn['answer'])
        with st.expander('Show retrieved passages & scores'):
            for it in turn['retrieved']:
                st.write(f"Source: {it['meta'].get('source', 'unknown')} | distance: {it.get('distance')} | rerank_score: {it.get('rerank_score', 'N/A')}")
                st.write(it['text'][:1000])
with col2:
    st.markdown('### Documents preview')
    # list collections
    try:
        from retriever import CHROMA_DIR
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=CHROMA_DIR))
        cols = client.list_collections()
        for c in cols:
            st.write(f"Collection: {c.name}")
            # show small sample
            coll = client.get_collection(c.name)
            try:
                docs = coll.get(include=['documents','metadatas'], limit=5)
                for d,m in zip(docs['documents'], docs['metadatas']):
                    st.write(m.get('source', 'unknown'))
                    st.write(d[:300])
            except Exception as e:
                st.write('preview error', e)
    except Exception as e:
        st.write('No DB yet or preview failed:', e)

    st.markdown('---')
    if st.button('Download transcript of chat'):
        lines = []
        for turn in st.session_state['history']:
            lines.append('Q: '+turn['query'])
            lines.append('A: '+(turn['answer'] or ''))
            lines.append('\n')
        txt = '\n'.join(lines)
        st.download_button('Click to download', txt, file_name='chat_transcript.txt')
