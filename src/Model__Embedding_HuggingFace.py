from langchain_community.embeddings import HuggingFaceEmbeddings as BaseEmbeddings
import sentence_transformers
class HuggingFaceEmbeddingsModel():
    def __init__(self):
        self.embeddings = BaseEmbeddings()

    def get_embedding(self, text):
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)