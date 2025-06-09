import os
from src.utils import get_relative_path
from src.engine.class_identifier.class_identifier import ClassIdentifier

from llama_index.core import Document, VectorStoreIndex, Settings, QueryBundle, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, fusion_retriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.huggingface import HuggingFaceLLM


class DescriptionBasedIdentifier(ClassIdentifier):
    def __init__(self, description_file_path: str, top_k: int):
        super().__init__(top_k)
        self.description_file_path = get_relative_path("./resources/" + description_file_path.split('/')[-1])
        absolute_path = os.path.abspath(self.description_file_path)
        raw_text = open(absolute_path, 'r').read()
        self._texts = raw_text.split("\n")
        self._documents = [Document(text=text) for text in self._texts]  
        self._text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=1)
        self.retriever = None
        # self.reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=5)
        
    def get_name(self):
        return "top-" + str(self.top_k)
    
    def get_resource(self):
        return self.description_file_path.split('/')[-1]
        

class SparseClassIdentifier(DescriptionBasedIdentifier):

    def __init__(self, description_file_path: str, top_k: int):    
        super().__init__(description_file_path, top_k)
            
        index_path = get_relative_path("./resources/indices/bm25_index_" + description_file_path.split('/')[-1])

        if os.path.exists(index_path):
            self.bm25_retriever = BM25Retriever.from_persist_dir(index_path)
        else:
            nodes = self._text_splitter.get_nodes_from_documents(self._documents, show_progress=False)
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)
            self.bm25_retriever.persist(index_path)
        
        self.retriever = self.bm25_retriever
        
    def get_name(self):
        return "sparse-" + super().get_name()


class DenseClassIdentifier(DescriptionBasedIdentifier):

    def __init__(self, description_file_path: str, top_k: int):     
        super().__init__(description_file_path, top_k)
                
        dense_index_path = get_relative_path("./resources/indices/dense_index_" + description_file_path.split('/')[-1])
        self.embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

        if os.path.exists(dense_index_path):
            storage_context = StorageContext.from_defaults(persist_dir=dense_index_path)
            self.vector_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            self.vector_index = VectorStoreIndex.from_documents(
                self._documents,
                show_progress=False,
                embed_model=self.embed_model,
                transformations=[self._text_splitter]
            )
            self.vector_index.storage_context.persist(dense_index_path)

        self.dense_retriever = VectorIndexRetriever(index=self.vector_index)   
        self.retriever = self.dense_retriever
    
    def get_name(self):
        return "dense-" + super().get_name()


QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates paraphrased search queries based on a "
    "single input query. Generate {num_queries} paraphrases, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

class HybridClassIdentifier(SparseClassIdentifier, DenseClassIdentifier):

    def __init__(self, description_file_path: str, top_k: int, fusion_mode: fusion_retriever.FUSION_MODES, llm_queries: int = 0):        
        super().__init__(description_file_path, top_k)
        
        self.llm_queries = llm_queries
        self.fusion_mode = fusion_mode
        
        if llm_queries == 0:
            Settings.llm = None
        else:
            Settings.llm = HuggingFaceLLM(model_name="google/gemma-3-4b-it", 
                                          tokenizer_name="google/gemma-3-4b-it", 
                                          context_window=4096)

        self.hybrid_retriever = QueryFusionRetriever(
            [
                self.dense_retriever,
                self.bm25_retriever
            ],
            mode=fusion_mode,
            num_queries=1 + llm_queries,
            use_async=False,
            query_gen_prompt=QUERY_GEN_PROMPT
        )
        
        self.retriever = self.hybrid_retriever
        
    def get_name(self):
        return "hybrid (" + self.fusion_mode + ")-" + super().get_name()
    
    def identify(self, question, top_k: int = None, threshold = 0, debug = False, return_labels = False):
        if top_k is not None:
            self.top_k = top_k
        self.bm25_retriever.similarity_top_k = self.top_k
        self.dense_retriever.similarity_top_k = self.top_k
        return super().identify(question, top_k, threshold, debug, return_labels)


if __name__ == '__main__':    
    identifier = SparseClassIdentifier(get_relative_path('./resources/wikidata_classes_20.txt'),
                                    top_k=5)
    print("How many people live in cities in the vicinity of the Nile ?")
    classes = identifier.identify(
        question="How many people live in cities in the vicinity of the Nile ?",
        debug=True)
    print(classes)
    
    
    identifier = DenseClassIdentifier(get_relative_path('./resources/wikidata_classes_20.txt'),
                                    top_k=5)
    print("How many people live in cities in the vicinity of the Nile ?")
    classes = identifier.identify(
        question="How many people live in cities in the vicinity of the Nile ?",
        debug=True)
    print(classes)
    
    
    identifier = HybridClassIdentifier(get_relative_path('./resources/wikidata_classes_20.txt'),
                                    top_k=5,
                                    fusion_mode=fusion_retriever.FUSION_MODES.RECIPROCAL_RANK,
                                    llm_queries=0)
    print("How many people live in cities in the vicinity of the Nile ?")
    classes = identifier.identify(
        question="How many people live in cities in the vicinity of the Nile ?",
        debug=True)
    print(classes)
    
