from typing import List, Tuple
from src.utils import get_relative_path
from src.engine.entity_linking.dev import NamedEntityRecognition
from src.utils import SupportedLLMs, llm_call
from src.engine.class_identifier.class_identifier import ClassIdentifier
from src.engine.class_identifier.description_based_identifiers import DenseClassIdentifier, DescriptionBasedIdentifier, SparseClassIdentifier
from src.engine.class_identifier.geoqa_concept_identifier import NgramClassIdentifier
import re


class ClassFilter:
    
    def __init__(self, model_id: str = SupportedLLMs.GEMMA3_12B):
        super().__init__()
        self.model_id = model_id
    
    def filter_relevant_classes(self, classes_with_labels: List[Tuple[str, str]], question: str, debug: bool = False, entities: List[str] = None) -> List[str]:        
        PROMPT = """
            You are an expert user of a Knowledge Graph. You are doing preliminary work on writing SPARQL queries that answer natural language questions/requests.
            Your are given a question/request and a list of class URIs and labels from the ontology that is employed by the Knowledge Graph. 
            Your job is to filter out the classes that are irrelevant to the question/request. 
            You are essentially asked to narrow down the list of classes to those that could be used to write a SPARQL query. If multiple classes refer to the same concept, include all of them. Don't skip duplicates.
            Please explain your reaosning for the selection.
            Your final answer must be a list of numbers from the given list. Surround your answer in square brackets and separate your chosen numbers with a comma.
            If none of the given classes are relevant, return an empty list.
            
            Question: {question}
            Classes: 
            {classes_with_labels}
            Answer:
        """
        
        PROMPT_ENTITIES = """
        A knowledge graph class or type refers to a category that defines the nature of some entities (nodes) in a knowledge graph. 
        For example, in a knowledge graph about movies, the class "Actor" would define the nature of entities that represent actors in that graph.
        So, "Angelina Jolie" would be an instance of the class "Actor", while "Titanic" would be an instance of the class "Movie".
        Classes are used to retrieve all the instances of that class from the knowledge graph.
        
        You are given a question, a list of nodes (entities) and an enumerated list of classes (types) from the ontology that is employed by the Knowledge Graph.
        Your task is to select the classes which will be used for retrieving an answer for the given question. Remember, classes are used to retrieve all the instances of that class from the knowledge graph.
        You must filter out classes which are irrelevant to the question or that are already fully covered by the entities.
        
        Example A:
            Question: "Which is the most popular book of J.R.R. Tolkien, other than the Lord of the Rings?"
            Entities: ["<http://example.org/J.R.R._Tolkien>", "<http://example.org/The_Lord_of_the_Rings>"]
            Classes:
            0. <http://example.org/book>
            1. <http://example.org/author>"
            2. <http://example.org/famous_british_authors>
        
        Expected answer: [0]
        
        Explanation:
        - The class "book" is relevant to the question and is required to find the answer. Using this we will retrieve all the books of J.R.R. Tolkien.
        - The class "author" is already covered by the entity "<http://example.org/J.R.R._Tolkien>". We don't need to retrieve any other authors, therefore we don't need this class to answer the question.
        - The class "famous_british_authors" is not immediately relevant to the question. We don't need any other famous british authors, so we don't need this class to answer the question.
        
        Example B:
            Question: "Show me the list of all human settlements in Tamriel."
            Entities: ["<http://example.org/Tamriel>"]
            Classes:
            0. <http://example.org/town>
            1. <http://example.org/human_fighter>
            2. <http://example.org/city>
            3. <http://example.org/region>
            4. <http://example.org/continent>
            5. <http://example.org/town_or_village>
            6. <http://example.org/village>
            
        Expected answer: [0, 2, 5, 6]
        
        Explanation:
        - The classes "0. town", "2. city" and "5. town_or_village", "6. village" are relevant to the question. To construct our answer we need to retrieve all the towns, cities, villages in Tamriel.
        - The classes "3. region" and "4. continent" are already covered by the entity "Temriel". We don't need any other regions or continents, therefore we don't need them to answer the question.
        - The class "1. human_fighter" is not relevant to the question and it is highly unlikely that it will be of use.
        
        As you can see your final answer must be a list of numbers from the given list. Surround your answer in square brackets and separate your chosen numbers with a comma. 
        If none of the given classes are relevant, return an empty list.
        
        Remember, your task is to identify the classes that are immediately relevant to the question and can be used to find the answer on the Knowledge graph.
        Remember, must filter out classes which are irrelevant to the question or that are already fully covered by the entities.
        
        **Think step-by-step and explain your reasoning before answering.**
            
        Question: {question}
        Entities: 
        {entities}
        Classes: 
        {classes_with_labels}
        
        Answer:
        """
        for attempt in range(3):
            classes_string = "\n".join(f"{idx}. {uri} - {label}" for idx, (uri, label) in enumerate(classes_with_labels))
            
            if entities is None:
                prompt = PROMPT.format(question=question, classes_with_labels=classes_string)
            else:
                entities_string = "\n".join("<http://example.org/" + entity.replace(" ", "_") + ">" for entity in entities)
                for entity in entities:
                    question.replace(entity, "<" + entity + ">")
                prompt = PROMPT_ENTITIES.format(question=question, classes_with_labels=classes_string, entities=entities_string)
            
            generated = llm_call(self.model_id, prompt, 4096)
            
            if debug:
                print(prompt)
                print(generated)
            
            selected_classes_string = re.findall(r"\[(.*?)\]", generated)
            if len(selected_classes_string) > 0:
                break
        
        classes_indices = []
        try:
            if len(selected_classes_string) > 0:
                classes_indices = [int(i) for i in selected_classes_string[-1].split(",")]
        except:
            print(selected_classes_string)
            classes_indices = range(len(classes_with_labels))
        
        classes = []
        for idx in classes_indices:
            try:
                classes.append(classes_with_labels[idx][0])
            except:
                continue
        
        return classes
    

class MixClassIdentifier(ClassIdentifier):

    def __init__(self, description_file_path: str, top_k: int):
        super().__init__(top_k)
        
        self.description_file_path = description_file_path
        self.sparse_identifier = SparseClassIdentifier(description_file_path,
                                                       top_k=int(top_k/3))
        self.dense_identifier = DenseClassIdentifier(description_file_path,
                                                     top_k=int(top_k/3))
        self.ngram_identifier = NgramClassIdentifier(description_file_path,
                                                     top_k=int(top_k/3))
        
        self.filter = ClassFilter(SupportedLLMs.GEMMA3_12B)
        
    def identify(self, question: str, top_k: int = None, threshold: float = 0.0, debug: bool = False, return_labels: bool = True, logging: bool = False):
        if top_k is not None:
            self.top_k = top_k
            
        sparse = self.sparse_identifier.identify(question, int(self.top_k/3), threshold, debug, return_labels)
        dense = self.dense_identifier.identify(question, int(self.top_k/3), threshold, debug, return_labels)
        ngram = self.ngram_identifier.identify(question, int(self.top_k/3), threshold, debug, return_labels)
        
        mixed = list(set(sparse + dense + ngram))
        
        if return_labels == False:
            if logging == False:
                return mixed
            else:
                return mixed, None
        
        filtered = self.filter.filter_relevant_classes(mixed, question, debug)

        if logging == False:
            return filtered
        else:
            return filtered, mixed
        
    def get_name(self):
        return "mixed-top-" + str(self.top_k)
    
    def get_resource(self):
        return self.description_file_path.split("/")[-1]
    
    
class MixClassIdentifier3(ClassIdentifier):

    def __init__(self, description_file_path: str, model_id: SupportedLLMs = SupportedLLMs.GPT4_1_MINI, top_k: int = 9):
        super().__init__(top_k)
        
        self.model_id = model_id
        self.description_file_path = description_file_path
        self.sparse_identifier = SparseClassIdentifier(description_file_path,
                                                       top_k=int(top_k))
        self.dense_identifier = DenseClassIdentifier(description_file_path,
                                                     top_k=int(top_k))
        self.ngram_identifier = NgramClassIdentifier(description_file_path,
                                                     top_k=int(top_k))
        
        self.ner = NamedEntityRecognition(model_id)
        self.filter = ClassFilter(model_id)
        
    def identify(self, question: str, top_k: int = None, threshold: float = 0.0, debug: bool = False, return_labels: bool = True, logging: bool = False):
        if top_k is not None:
            self.top_k = top_k
            
        sparse = self.sparse_identifier.identify(question, int(self.top_k), threshold, debug, return_labels)
        dense = self.dense_identifier.identify(question, int(self.top_k), threshold, debug, return_labels)
        ngram = self.ngram_identifier.identify(question, int(self.top_k), threshold, debug, return_labels)
        
        
        lists = [ngram, sparse, dense]
        candidates = []
        while True:
            max_len = max([len(l) for l in lists])
            if max_len == 0:
                break
            for l in lists:
                if len(l) == max_len and len(l) > 0:
                    candidates.append(l.pop(0))
                    candidates = list(set(candidates)) 
                    break
            if len(candidates) == self.top_k:
                break         
            
        # mixed = list(set(sparse + dense + ngram))
        
         # Remove duplicates
        
        if return_labels == False:
            if logging == False:
                return candidates
            else:
                return candidates, None
        
        entities = self.ner.ner(question, debug=debug)
        
        filtered = self.filter.filter_relevant_classes(candidates, question, debug, entities)
        
        filtered = list(set(filtered))  # Remove duplicates
        print(f"Filtered classes: {len(filtered)}")

        if logging == False:
            return filtered
        else:
            return filtered, {"entities": entities, "candidates": candidates}
        
    def get_name(self):
        return "dev-filtered" + str(self.top_k)
    
    def get_resource(self):
        return self.description_file_path.split("/")[-1]
    

if __name__ == '__main__':                
    identifier = MixClassIdentifier3(get_relative_path('./resources/wikidata_classes_20.txt'), model_id=SupportedLLMs.VLLM, top_k=10)
    print("How many people live in New York City ?")
    classes = identifier.identify("How many people live in New York City ?", 15, debug=True, return_labels=True)
    # print("MIXED")
    for c in classes:
        print(str(c))