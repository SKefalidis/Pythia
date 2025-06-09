from typing import List
import os
from llama_index.core.node_parser import SentenceSplitter
from src.utils import get_relative_path
from src.engine.class_identifier.class_identifier import ClassIdentifier
import requests
import json
from nltk.util import ngrams as create_ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import jellyfish
from llama_index.core import Document
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class NgramSimilarityRetriever(BaseRetriever):
    def __init__(self, class_dictionary_file_path: str, top_k: int = 0, callback_manager = None, object_map = None, objects = None, verbose = False):
        super().__init__(callback_manager, object_map, objects, verbose)
        
        self.top_k = top_k
        self.description_file_path = get_relative_path("./resources/" + class_dictionary_file_path.split('/')[-1])
        absolute_path = os.path.abspath(self.description_file_path)
        raw_text = open(absolute_path, 'r').read()
        self._texts = raw_text.split("\n")
        self._documents = [Document(text=text) for text in self._texts]  
        self._text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=1)
        self.nodes = self._text_splitter.get_nodes_from_documents(self._documents, show_progress=False)   
        
        self.labelToClassMap = []
        for node in self.nodes:
            uri, label = node.text.split(" - ")[0], " ".join(node.text.split(" - ")[1:]) # handling the case where the label contains the separator
            self.labelToClassMap.append((label, uri.strip(), node))
        
        self.removeNearInstances = False
        self.jwSimilarity = 0.98    
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def isSimilar(self, str1: str, str2: str, similarity_function) -> bool:
        if similarity_function == "jw":
            return jellyfish.jaro_winkler_similarity(str1, str2)
        elif similarity_function == "levenshtein":
            return jellyfish.jaro_winkler_similarity(str1, str2)
        else:
            raise ValueError("Unknown similarity function: {}".format(similarity_function))

    def _retrieve(self, query_bundle: str) -> List[NodeWithScore]:
        question = query_bundle.query_str
        
        nodes: List[NodeWithScore] = []
        
        # tokens = word_tokenize(question)
        # tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        # print(tokens)
        
        for conceptLabel, uri, node in self.labelToClassMap:
            tokens = word_tokenize(question)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in self.stop_words]
            ngrams = list(create_ngrams(tokens, len(conceptLabel.split())))
            
            for ngram in ngrams:
                similarity = 0
                
                if "sentinel" in conceptLabel:
                    similarity = ' '.join(ngram).lower() == conceptLabel.lower()
                else:
                    similarity = max(self.isSimilar(' '.join(ngram), conceptLabel.lower(), "jw"), self.isSimilar(' '.join(ngram), conceptLabel.lower() + "s", "jw"))
                    
                nodes.append(NodeWithScore(node=node, score=similarity))
        
        if self.top_k > 0:
            nodes.sort(key=lambda node: node.score, reverse=True)
            return nodes[:self.top_k]
        else: # FIXME: almost like GeoQA, the only problem is that we can have multiple concepts for the same node
            filtered_nodes = []
            for n in nodes:
                if n.score >= self.jwSimilarity:
                    filtered_nodes.append(n)
            return filtered_nodes


class NgramClassIdentifier(ClassIdentifier):
    
    def __init__(self, class_dictionary_file_path: str, top_k: int):
        super().__init__(top_k)
        self.class_dictionary_file_path = class_dictionary_file_path
        self.retriever = NgramSimilarityRetriever(class_dictionary_file_path, top_k)   

    def geoqa_send_request(self, question: str, url: str) -> requests.Response:
        data = {
            "question": question,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response
    
    def api_identify(self, question: str):
        response = self.geoqa_send_request(
            question, 'http://localhost:12345/startquestionansweringwithtextquestion').json()
        if response['status'] == 200:
            concepts = response['concepts']
            return concepts
        else:
            return []
        
    @staticmethod
    def get_name():
        return "ngram"
    
    def get_resource(self):
        return self.class_dictionary_file_path.split('/')[-1]


if __name__ == '__main__':
    from tqdm import tqdm
    
    identifier = NgramClassIdentifier(get_relative_path('./resources/wikidata_classes_20.txt'), 5)
    print("How many people live in cities in the vicinity of the Nile ?")
    classes = identifier.identify(question="How many people live in cities in the vicinity of the Nile ?")
    for c in classes:
        print(str(c))
