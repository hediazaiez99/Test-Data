from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === Charger le PDF ===
pdf_path = "CELEX_52013XC0802(04)_EN_TXT (1) GUIDELINE.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# === Chunker le texte ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# === Embeddings locaux avec sentence-transformers ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_local_guideline"
)
vectordb.persist()

# === LLM local pour répondre (modèle HF) ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # ou "tiiuae/falcon-7b-instruct" etc.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# === QA avec récupération locale ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# === Détection des questions nécessitant un pays ===
def needs_country_context(question):
    keywords = ["batch size", "fold", "increase", "scale up"]
    return any(kw in question.lower() for kw in keywords)

# === Interaction utilisateur ===
while True:
    question = input("\n❓ Your question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    if needs_country_context(question):
        country = input("🌍 Please specify the country or procedure (e.g. France, centralised, MRP, national): ")
        full_question = f"In the context of {country}, {question}"
    else:
        full_question = question

    response = qa.run(full_question)
    print("\n💬", response)
