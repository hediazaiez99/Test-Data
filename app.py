import re
import gradio as gr
from ctransformers import AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Chargement du PDF ===
pdf_path = "CELEX_52013XC0802(04)_EN_TXT (1) GUIDELINE.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
annex_chunks = [doc for doc in chunks if "ANNEX" in doc.page_content.upper() or "A. ADMINISTRATIVE CHANGES" in doc.page_content.upper()]

# === Embedding + ChromaDB ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_full")
annex_vectordb = Chroma.from_documents(annex_chunks, embedding=embedding_model, persist_directory="./chroma_annex")
retriever = vectordb.as_retriever()
annex_retriever = annex_vectordb.as_retriever()

# === Chargement du modèle Mistral quantisé via ctransformers avec réglages anti-boucle ===
mistral_llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    max_new_tokens=512,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
)

# === Génération de réponse experte
def generate_expert_response_mistral(context, question):
    prompt = f"""
### Instruction:
You are an EU regulatory expert. Read the context and answer the question clearly. Avoid repeating words. Summarize and infer where needed.

### Context:
{context}

### Question:
{question}

### Answer:
"""
    response = mistral_llm(prompt)
    if "change of a change of" in response.lower():
        return "❌ La réponse générée semble incohérente. Essaie une autre formulation ou question plus précise."
    return response.strip()

# === Lecture dynamique des séquences 1,2,3
def explain_sequence_from_annex(sequence):
    steps = re.findall(r"\d+", sequence)
    output = []
    for step in steps:
        results = annex_retriever.invoke(f"step {step}")
        if results:
            first_line = results[0].page_content.strip().split("\n")[0]
            output.append(f"**{step}** → {first_line}")
        else:
            output.append(f"**{step}** → ❓ Not found")
    return "### 📋 Interpreted Steps\n" + "\n".join(output)

# === Fonction principale du chatbot
def smart_chat(user_input, history):
    try:
        if re.search(r"\d+(?:,\s*\d+)+", user_input):
            return explain_sequence_from_annex(user_input)

        relevant_docs = retriever.invoke(user_input)
        if not relevant_docs:
            return "⚠️ Aucun contenu pertinent trouvé dans le document."

        context = "\n".join([doc.page_content for doc in relevant_docs[:3]])
        return generate_expert_response_mistral(context, user_input)

    except Exception as e:
        return f"❌ Erreur : {str(e)}"

# === Interface Gradio
gr.ChatInterface(
    smart_chat,
    title="💊 Smart EU RAG Chatbot (Mistral via CTransformers)",
    description="Pose tes questions réglementaires ou entre une séquence comme 1,2,3 pour voir l’interprétation depuis l’annexe."
).launch(share=True)
