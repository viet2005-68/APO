from FlagEmbedding import BGEM3FlagModel

class Embedder:
    def __init__(self):
        self.embedding_model = BGEM3FlagModel('BAAI/bge-m3')
    
    def embed_texts(self, texts: list):
        return self.embedding_model.encode(
            texts,
            batch_size=8,
            max_length=128
        )["dense_vecs"]