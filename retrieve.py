from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")

load_vector_store = Chroma(persist_directory="stores/insurance_cosine", embedding_function=embeddings)

query = "What is Group life insurance?"

docs = load_vector_store.similarity_search_with_score(query=query, k=3)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})