from typing import List
from src.utils import embed_model
from src.datasets.dataset import Dataset
from src.datasets.qald9_dataset import Qald9Dataset

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode

class QueryDb:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        
        Settings.llm = None

        # Build documents (nodes) for each query
        nodes = []
        for i in self.dataset:
            question = self.dataset.get_question(i)
            query = self.dataset.get_query(i)
            # We'll use the question as the text to embed, but store query as metadata
            node = TextNode(text=question, metadata={"query": query})
            nodes.append(node)

        # Build vector index
        self.index = VectorStoreIndex(nodes, embed_model=embed_model)
        self.retriever = VectorIndexRetriever(index=self.index)   

    def get_relevant_queries(self, input_question: str, top_k: int = 3) -> List[str]:
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        results = query_engine.query(input_question)

        # Extract queries from metadata
        relevant_questions = [node.text for node in results.source_nodes]
        relevant_queries = [node.metadata["query"] for node in results.source_nodes]
        return relevant_questions, relevant_queries
        
        
if __name__ == '__main__':
    dataset = Qald9Dataset.from_files("PATH_TO_DATASET_FILE")
    query_db = QueryDb(dataset)
    
    queries = query_db.get_relevant_queries("What is the capital of France?", top_k=5)
    for q in queries:
        print(q)