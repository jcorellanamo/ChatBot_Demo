"""
chatbotCPP.py  –  Streamlit + LangChain
Asistente para consultar artículos y penas del Código Procesal Penal Chileno
────────────────────────────────────────────────────────────────────────────
Ejecución ⇒
C:/Users/jcore/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m streamlit run chatbotCPP.py
"""

# ─── Imports ─────────────────────────────────────────────────────────────
import os, re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# ─── Configuración básica ────────────────────────────────────────────────
PDF_FILE  = "codigo_procesal_penal.pdf"
INDEX_DIR = "codigo_procesal_penal_idx"

st.set_page_config(page_title="Asistente Código Procesal Penal-Demo", page_icon="⚖️")
st.title("⚖️ Asistente Virtual del Código Procesal Penal (Demo)")

# ─── A. Splitter por artículo ────────────────────────────────────────────
class PorArticuloSplitter(TextSplitter):
    pattern = re.compile(r"(?=Artículo\s+\d+\.)", flags=re.IGNORECASE)
    def split_text(self, text: str):
        partes, trozos = self.pattern.split(text), []
        for i in range(1, len(partes), 2):
            cab = partes[i]
            cuerpo = partes[i + 1] if i + 1 < len(partes) else ""
            chunk = (cab + cuerpo).strip()
            if chunk:
                trozos.append(chunk)
        return trozos

def cargar_y_dividir(pdf_path: str):
    texto = "\n".join(p.page_content for p in PyPDFLoader(pdf_path).load())
    splitter  = PorArticuloSplitter(chunk_overlap=0)
    articulos = splitter.split_text(texto)
    return [Document(page_content=a, metadata={"source": f"Art. {i+1}"})
            for i, a in enumerate(articulos)]

# ─── B. Embeddings + FAISS ───────────────────────────────────────────────
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

if os.path.exists(INDEX_DIR):
    vectordb = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
else:
    if not os.path.exists(PDF_FILE):
        st.error(f"No se encontró {PDF_FILE} en la carpeta actual.")
        st.stop()
    docs = cargar_y_dividir(PDF_FILE)
    with st.expander("🔍 Ejemplos de chunks"):
        for d in docs[:3]:
            st.code(d.page_content[:400] + "…")
    vectordb = FAISS.from_documents(docs, emb)
    vectordb.save_local(INDEX_DIR)

# ─── C. Retriever (k=20 + opcional Cohere) ───────────────────────────────
base_retriever = vectordb.as_retriever(search_kwargs={"k": 20})
retriever = base_retriever

if "COHERE_API_KEY" in st.secrets:
    try:
        from langchain_community.document_compressors import CohereRerank
    except ImportError:
        try:
            from langchain_community.document_compressors.cohere_rerank import CohereRerank
        except ImportError:
            CohereRerank = None
    if CohereRerank:
        retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=CohereRerank(model="rerank-multilingual-v3.0", top_n=5),
        )

# ─── D. LLM & chain ──────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="learnlm-2.0-flash-experimental",
    temperature=0,
    api_key=st.secrets["GOOGLE_API_KEY"],
)

system_prompt = (
    "A continuación tienes fragmentos del Código Procesal Penal. "
    "Responde indicando la pena y el artículo correspondiente. "
    "Si la información no aparece, responde: «No se encontró en el texto». "
    "Contexto: {context}"
)
prompt   = ChatPromptTemplate.from_messages([("system", system_prompt),
                                             ("human", "{input}")])
qa_chain = create_stuff_documents_chain(llm, prompt)
chain    = create_retrieval_chain(retriever, qa_chain)

# ─── E. Estado de sesión ────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

# ─── F. Placeholder del historial lateral ───────────────────────────────
history_box = st.sidebar.empty()          # un contenedor vacío en el sidebar

def render_history():
    with history_box.container():
        st.header("🗂️ Historial de consultas")
        if st.session_state["history"]:
            for i, q in enumerate(reversed(st.session_state["history"]), 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.caption("Aún no hay consultas registradas.")

# primera pinta (por si ya hay algo)
render_history()

# ─── G. Chat principal ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

consulta = st.chat_input("Escribe tu consulta y presiona Enter…")

if consulta:
    # Guarda en historial ANTES de redibujar
    st.session_state["history"].append(consulta)

    # Muestra mensaje del usuario
    with st.chat_message("user"):
        st.markdown(consulta)
    st.session_state.messages.append({"role": "user", "content": consulta})

    # Genera respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando…"):
            try:
                resultado  = chain.invoke({"input": consulta})
                respuesta  = resultado["answer"].strip()
            except Exception as e:
                respuesta = f"Error: {e}"
        st.markdown(respuesta)

        with st.expander("📑 Fragmentos utilizados"):
            for frag in resultado.get("context", []):
                st.write(f"**{frag.metadata.get('source','?')}**")
                st.code(frag.page_content[:600] + "…")

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

    # ¡Redibuja el historial YA con la nueva pregunta!
    render_history()
