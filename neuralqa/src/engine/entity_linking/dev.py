from ast import Load
from typing import List

from jellyfish import jaro_winkler_similarity
from src.utils import SupportedLLMs, get_relative_path, llm_call, embed_model, load_faiss_index, search_faiss_index
from src.datasets.dataset import KnowledgeGraph
from src.engine.entity_linking.entity_linker import EntityLinker
import re
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from llama_index.core import Document, VectorStoreIndex, Settings, QueryBundle, StorageContext, ServiceContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, fusion_retriever
from llama_index.retrievers.bm25 import BM25Retriever
from sentence_transformers.util import cos_sim


class NamedEntityRecognition():
    
    PROMPT = """A named entity in Named Entity Recognition (NER) is an object, real or fictional, that can be identified with a proper name or Universally Unique Identifier. 
    For example, these entities can be names of people, organizations, locations, specific terms, constructs, files, resources, ideas that carry meaning in context.
    
    For example a book in general is not a named entity, but the book "The Hitchhiker's Guide to the Galaxy" is a named entity.
    A file in general is not a named entity, but the file "my_file.txt" is a named entity.
    A person in general is not a named entity, but the person "Kyriakos Grizzly" is a named entity.
    
    Your task is to identify all named entities in the given sentence, even if you do not know the exact entity that the sentence is referring to. 
    For this task, do not consider quantities, dates, coordinates and other such values as Named Entities.
    If something is in parentheses think twice before deciding that it is not a named entity.
    If the entity is an abbreviation, you should also include both the full name and the abbreviation in your results.
    You must put your Named Entities inside {{ }}.
    
    Here are some examples of inputs and their named entities:
    INPUT: Who is Kyriakos Grizzly?
    OUTPUT: {{Kyriakos Grizzly}}
    INPUT: How many cases of measles did citizens of Greece get in 2022?
    OUTPUT: {{Greece}}
    INPUT: Show me all files that reference IDX-1536 and have been created after Sergios Kefalidinson won his first olympic medal.
    OUTPUT: {{IDX-1536}} {{Sergios Kefalidinson}}
    
    Remember, a named entity in Named Entity Recognition (NER) is an object, real or fictional, that can be identified with a proper name or Universally Unique Identifier. 
    Remember, your task is to identify all named entities in the given sentence, even if you do not know the exact entity that the sentence is referring to.
    You must differentiate between named entities and concepts, ideas, and other general terms. A named entity is a specific, named, identifiable object, that can be referenced by a proper name or a unique identifier.
    Ignore capitalization, punctuation, and other such details. A capitalized word is not necessarily a named entity, and a lowercase word can be a named entity. Do not consider capitalization as a factor in your decision.
    Users are unpredictable and might not be writing in proper English. 
    Focus on the meaning of the words and their context.
    
    Think step by step, explain what the sentence is about, and then identify the named entities in the sentence.
    
    INPUT: {sentence} 
    OUTPUT: """

    def __init__(self, model_id: SupportedLLMs = SupportedLLMs.GEMMA3_12B):
        super().__init__()
        self.model_id = model_id

    def ner(self, sentence: str, debug: bool = False, logging: bool = False):
        prompt = NamedEntityRecognition.PROMPT.format(
            sentence=sentence)
        generated = llm_call(self.model_id, prompt, 4096)
        
        if debug:
            print(prompt)
            print(generated)
            print()
        
        pat = r'(?<=\{).+?(?=\})'
        entities = re.findall(pat, generated)
        entities = list(set(entities))  # Remove duplicates
        # print(entities)
        
        if logging == False:
            return entities
        else:
            return entities, [("ner_generated", generated), ("ner_results", entities)]


class NamedEntityDisambiguation():
    
    PROMPT_TASK = """
    You are given a natural language question, a named entity that has been identified in that question and an enumerated list of URIs, possibly with their labels and descriptions from a Knowledge Graph. 
    Decide which URI is more appropriate for the given named entity in the context of the question.
    To answer select a number. You must put the number inside { }."""
    
    PROMPT_TOOLS_DEFINITION = """
    In addition to selecting a candidate entity, you can also use tools to help you in your task. When using a tool you must wrap the function call in { }. You can call only one function per response.`.
    
    The following tools are available:
        def get_more_candidates() -> List[str]:
            \"\"\" Retrieves and returns additional candidates for the given named entity.
            \"\"\"
        
        def get_predicates(candidate: int) -> List[str]:
            \"\"\" Retrieves and returns all predicates for the specified candidate entity.
            \"\"\"
            
        def get_type(candidate: int) -> List[str]:
            \"\"\" Retrieves and returns ontology types/classes of the specified candidate entity.
            \"\"\"
            
        def get_most_similar_predicate(search_query: str) -> List[str]:
            \"\"\" Retrieves and returns the most similar predicates of each candidate for the given search query.
            \"\"\"
            
        def get_candidate_popularity() -> List[str]:
            \"\"\" Retrieves and returns the amount of triples for each candidate entity. This is a good indicator of the popularity of the candidate in the Knowledge Graph.
            \"\"\"
    
    If you are certain that none of the candidates are appropriate, use `get_more_candidates()` to retrieve additional candidates.
    If you have a preferred candidate, but want to verify your choice use `get_predicates(candidate)` and/or `get_type(candidates)` to get information about the predicates and type for the specified candidate.
    If you want to learn about relevant predicates of each candidate, use `get_most_similar_predicate(search_query)` to get the most similar predicates of each candidate for the given search query.
    If you are still uncertain, you can use `get_candidate_popularity()` to get the amount of triples for each candidate entity. This is a good indicator of the popularity of the candidate in the Knowledge Graph.
    
    Remember, you can call only one function per response. Remember When using a tool you must wrap the function call in { }.
    
    After the information is retrieved, you can use it to make a better decision. Do not be afraid to use the tools to help you in your task. You will have the chance to use them again if you need to."""
    
    PROMPT_EXAMPLES = """
    Here are some examples (I skip the reasoning explanation part, but you should reason and explain your answer):
    INPUT: Kyriakos Mitsotakis 
    0. http://knowledge.com/Kostas_Mitsotakis
    1. http://knowledge.com/Kyriakos_Grizzly
    2. http://knowledge.com/Kyriakos_Mitsotakis
    OUTPUT: {2}
    
    INPUT: Big Apple 
    0. https://en.wikipedia.org/wiki/Apple_Inc.
    1. https://en.wikipedia.org/wiki/New_York_City
    2. https://en.wikipedia.org/wiki/Big_Apple
    3. https://en.wikipedia.org/wiki/Applebee%27s
    OUTPUT: {1}"""
    
    PROMPT_TOOL_USE_EXAMPLES = """
    INPUT: smark 
    0. https://kg.gr/stefanos_markopoulos
    1. https://kg.gr/marking_id
    2. https://kg.gr/markos
    OUTPUT: {get_predicates(0)}
    
    In this case I chose to get additional information about the first candidate, since it is the most promising one but I am not completely certain.
    
    INPUT: For the request 'What is the surface area of the municipality of Athens?' I have identified the named entity 'municipality of Athens' for you to disambiguate. 
    0. https://geo-kg.org/Athens
    1. https://geo-kg.org/Ancient_Athens
    2. https://geo-kg.org/Athens_2025
    OUTPUT: {get_most_similar_predicate("geometry")}
    
    In this case I want to check which candidates have spatial information before making a decision. Alternatively, I could use get_predicates() to get all predicates for the most likely candidate."""
    
    PROMPT_REASONING = """
    Think about your answer and explain your decision. You must first think and then give your response."""
    
    PROMPT_TEMPLATE = """
    {prompt_task}
    
    {prompt_tools_definition}
        
    {prompt_examples}
    
    {prompt_tool_use_examples}

    {prompt_reasoning}
    
    INPUT:
    For the request '{request}' I have identified the named entity '{entity}' for you to disambiguate.
    CANDIDATES:
    {uris}
    OUTPUT:
    """

    def __init__(self, model_id: SupportedLLMs = SupportedLLMs.GEMMA3_12B, dense_index_name: str = None, sparse_index_name: str = None):
        super().__init__()
        self.model_id = model_id
        
        if dense_index_name is not None and dense_index_name != "":
            self.has_dense_index = True
            dense_index_path = get_relative_path("./resources/indices/" + dense_index_name)

            if os.path.exists(dense_index_path):
                print("Loading index from: " + dense_index_path)
                start = time.time()
                
                self.use_faiss = "faiss" in dense_index_path
                
                if self.use_faiss:
                    # FAISS
                    self.faiss_index, self.documents = load_faiss_index(dense_index_path)
                else:
                    # built-in Vector Store
                    storage_context = StorageContext.from_defaults(persist_dir=dense_index_path)
                    self.vector_index = load_index_from_storage(storage_context, embed_model=embed_model)
                    self.retriever = VectorIndexRetriever(index=self.vector_index,
                                                        similarity_top_k=5)
                
                print("Index loaded in: " + str(time.time() - start) + " seconds")
            else:
                print("[Error] Index not found!")
        else:
            self.has_dense_index = False
            
        if sparse_index_name is not None and sparse_index_name != "":
            self.has_sparse_index = True
            sparse_index_path = get_relative_path("./resources/indices/" + sparse_index_name)
            if os.path.exists(sparse_index_path):
                print("Loading index from: " + sparse_index_path)
                start = time.time()
                
                self.bm25_retriever = BM25Retriever.from_persist_dir(sparse_index_path)
                
                print("Index loaded in: " + str(time.time() - start) + " seconds")
            else:
                print("[Error] Index not found!")
        else:
            self.has_sparse_index = False
            
    def _get_sparse_candidates(self, entity: str, k: int = 5, debug = False):
        try:
            self.bm25_retriever.similarity_top_k = k
            response = self.bm25_retriever.retrieve(entity)           
        except Exception as e:
            print(entity)
            print(e)
        nodes = [node for node in response]
        candidates = [node.get_text() for node in nodes]
        if debug:
            print("Entity:", entity)
            print("Response:")
            for node in nodes:
                print(node)
                print(node.score)
        return candidates
            
    def _get_dense_candidates(self, entity: str, k: int = 5, debug = False):
        if self.use_faiss:
            candidates = search_faiss_index(self.faiss_index, self.documents, entity, k=k, debug=debug)
        else:
            try:
                response = self.retriever.retrieve(entity)           
            except Exception as e:
                print(entity)
                print(e)
            nodes = [node for node in response]
            candidates = [node.get_text() for node in nodes]
            if debug:
                print("Entity:", entity)
                print("Response:")
                for node in nodes:
                    print(node)
                    print(node.score)
        # Remove embedding prefix
        candidates = list(map(lambda x: x.replace('search_document: ', ''), candidates))
        return candidates

    def discover_candidates(self, entity: str, k: int = 5, debug = False):
        sparse_candidates = self._get_sparse_candidates(entity, k=k, debug=debug) if self.has_sparse_index else []
        dense_candidates = self._get_dense_candidates(entity, k=k, debug=debug) if self.has_dense_index else []
        candidates = []
        while len(candidates) < k:
            if len(sparse_candidates) > 0 and len(sparse_candidates) > len(dense_candidates):
                if sparse_candidates[0] not in candidates:
                    candidates.append(sparse_candidates.pop(0))
                else:
                    sparse_candidates.pop(0)
            elif len(dense_candidates) > 0:
                if dense_candidates[0] not in candidates:
                    candidates.append(dense_candidates.pop(0))
                else:
                    dense_candidates.pop(0)
        return candidates
    
    def tool_get_more_candidates(self, entity: str, k: int = 10, start: int = 0, end: int = -1):
        candidates = self.discover_candidates(entity, k=k)[start:end]
        new_candidates_string = "\n\t".join(f"{idx}. {candidate}" for idx, candidate in enumerate(candidates))
        PROMPT = f"""
        These are the new disambiguation candidates for {entity}:
        
        {new_candidates_string}
        
        No other candidates are available."""
        return candidates, PROMPT
    
    def _get_predicates_for_entity(self, entity: str, endpoint: str, limit: int = 20):
        QUERY = f"""
            SELECT ?p (COUNT (?o) as ?c) WHERE {{
                <{entity}> ?p ?o .
            }}
            GROUP BY ?p
            ORDER BY DESC(?c)
            LIMIT {limit}
        """
        sparql = SPARQLWrapper(endpoint)
        sparql.setCredentials("user", "PASSWORD")
        sparql.setQuery(QUERY)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Error querying SPARQL: {e}")
            print(f"Query: {QUERY}")
            return ['Error during processing, no predicates found.']
        predicates = []
        for result in results["results"]["bindings"]:
            predicates.append(result["p"]["value"])
        return predicates
    
    def tool_get_predicates(self, entity: str, endpoint: str):        
        predicates = self._get_predicates_for_entity(entity, endpoint)
        predicates_string = "\n\t".join(f"{idx}. {predicate}" for idx, predicate in enumerate(predicates))
        PROMPT = f"""
        These are the {len(predicates)} most common predicates for the entity {entity}:
        
        {predicates_string}
        
        If you want to search for specific predicates, you can use the function get_most_similar_predicate(search_query) to get the most similar predicates of each candidate for the given search query."""
        return PROMPT
    
    def _get_type_for_entity(self, entity: str, endpoint: str):
        # FIXME: Does not get the type correctly for Wikidata.
        QUERY = f"""
            SELECT ?type WHERE {{
                <{entity}> a ?type .
            }}
        """
        sparql = SPARQLWrapper(endpoint)
        sparql.setCredentials("user", "PASSWORD")
        sparql.setQuery(QUERY)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Error querying SPARQL: {e}")
            print(f"Query: {QUERY}")
            return ['Error during processing, no predicates found.']
        types = []
        for result in results["results"]["bindings"]:
            types.append(result["type"]["value"])
        return types
    
    def tool_get_type(self, entity: str, endpoint: str):        
        types = self._get_type_for_entity(entity, endpoint)
        types_string = "\n\t".join(f"{idx}. {type}" for idx, type in enumerate(types))
        PROMPT = f"""
        These are the types/classes for the entity {entity}:
        
        {types_string}"""
        return PROMPT
    
    def tool_get_most_similar_predicate(self, search_query: str, uris: List[str], endpoint: str):
        most_similar_predicates = []
        for uri in uris:
            predicates = self._get_predicates_for_entity(uri, endpoint, 100)
            # most_similiar_predicate = max(predicates, key=lambda x: jaro_winkler_similarity(search_query, x))
            top2_most_similar_predicate = sorted(predicates, key=lambda p: cos_sim(embed_model.get_text_embedding(search_query), embed_model.get_text_embedding(p)), reverse=True)[:2]
            most_similar_predicates.append((uri, ", ".join(top2_most_similar_predicate)))
        most_similar_predicates_string = "\n\t".join(f"{idx}. {uri} - {predicates}" for idx, (uri, predicates) in enumerate(most_similar_predicates))
        PROMPT = f"""
        These are the top-2 most similar predicates for each candidate for the search query '{search_query}':
        
        {most_similar_predicates_string}"""
        return PROMPT
    
    def _get_popularity_for_entity(self, entity: str, endpoint: str):
        if entity == "":
            return 0
        
        QUERY = f"""
            SELECT (COUNT (*) as ?c) WHERE {{
                {{ <{entity}> ?p ?o . }}
                UNION
                {{ ?s ?p <{entity}> . }}
            }}
        """
        sparql = SPARQLWrapper(endpoint)
        sparql.setCredentials("user", "PASSWORD")
        sparql.setQuery(QUERY)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"Error querying SPARQL: {e}")
            print(f"Query: {QUERY}")
            return 0
        popularity = 0
        for result in results["results"]["bindings"]:
            popularity = (result["c"]["value"])
        return popularity
    
    def tool_get_candidate_popularity(self, candidates: List[str], endpoint: str):
        candidate_popularity = []
        for candidate in candidates:
            popularity = self._get_popularity_for_entity(candidate, endpoint)
            candidate_popularity.append((candidate, popularity))
        candidate_popularity_string = "\n\t".join(f"{idx}. {candidate} - {popularity}" for idx, (candidate, popularity) in enumerate(candidate_popularity))
        PROMPT = f"""
        These are the popularity scores for each candidate:
        
        {candidate_popularity_string}"""
        return PROMPT

    def ned(self, question: str, entity: str, endpoint: str = None, debug: bool = False, logging: bool = False):
        if debug:
            print("Entity: " + entity)

        k = 10
        candidates = self.discover_candidates(entity, debug=debug, k=k)
        candidates_string = "\n\t".join(f"{idx}. {candidate}" for idx, candidate in enumerate(candidates))
        
        print(f"[DEBUG] Found {len(candidates)} candidates for entity '{entity}': {candidates_string}")
        
        prompt = NamedEntityDisambiguation.PROMPT_TEMPLATE.format(
            request=question,
            entity=entity,
            uris=candidates_string,
            prompt_task=NamedEntityDisambiguation.PROMPT_TASK,
            prompt_tools_definition=NamedEntityDisambiguation.PROMPT_TOOLS_DEFINITION,
            prompt_examples=NamedEntityDisambiguation.PROMPT_EXAMPLES,
            prompt_tool_use_examples=NamedEntityDisambiguation.PROMPT_TOOL_USE_EXAMPLES,
            prompt_reasoning=NamedEntityDisambiguation.PROMPT_REASONING)
        
        all_generated = []
        while True:
            # ----- Generate -----
            if debug:
                print("[DEBUG] PROMPT:" + prompt + "\n")
            generated = llm_call(self.model_id, prompt, 2048)
            if debug:
                print("[DEBUG] GENERATED:" + generated + "\n")
            all_generated.append(generated)
            
            # ----- Parse response -----
            pat = r'(?<=\{).+?(?=\})'
            response = re.findall(pat, generated)
            try:
                if debug:
                    print("[DEBUG] RESPONSE:")
                    print(response)
                if "get_more_candidates" in response[0]:
                    candidates, response = self.tool_get_more_candidates(entity, k=k*2, start=k, end=-1)
                elif "get_predicates" in response[0]:
                    entity_id = response[-1].replace("get_predicates(", "").replace(")", "")
                    entity = candidates[int(entity_id)].split(', ')[0]
                    response = self.tool_get_predicates(entity, endpoint)
                elif "get_type" in response[0]:
                    entity_id = response[-1].replace("get_type(", "").replace(")", "")
                    entity = candidates[int(entity_id)].split(', ')[0]
                    response = self.tool_get_type(entity, endpoint)
                elif "get_most_similar_predicate" in response[0]:
                    search_query = response[-1].replace("get_most_similar_predicate(", "").replace(")", "")
                    response = self.tool_get_most_similar_predicate(search_query, [c.split(", ")[0] for c in candidates], endpoint)
                elif "get_candidate_popularity" in response[0]:
                    response = self.tool_get_candidate_popularity([c.split(", ")[0] for c in candidates], endpoint)
                else:
                    entities = response
                    break
            except Exception as e:
                print("[ERROR] Exception: " + str(e))
                print("[ERROR] Full generation: " + generated)
                print("[ERROR] Likely tool failure. Skipping tool usage.")
                entities = []
                break
            
            # ----- Prepare prompt for next iteration -----
            prompt = prompt + "\n" + generated +"\n" + response
            
        results = None
        if len(entities) > 0:
            try:
                index = int(entities[0])
                results = candidates[index].split(', ')[0]
            except:
                results = ""
        elif len(entities) == 0 and 'get_more_candidates' in generated:
            results = ""
        elif len(entities) == 0 and len(candidates) > 0:
            results = candidates[0].split(', ')[0]
        else:
            results = ""
        
        if logging == False:
            return results
        else:
            return results, {"ned_entity": entity, "ned_candidates": candidates, "ned_generated": prompt + " #GENERATION# " + generated, "ned_results": results}


class DevLinker(EntityLinker):

    def __init__(self, knowledge_graph: KnowledgeGraph, model: SupportedLLMs = SupportedLLMs.GPT4_1_MINI, dense_index_name: str = None, sparse_index_name: str = None):
        super().__init__(knowledge_graph)
        self.endpoint = KnowledgeGraph.get_endpoint(knowledge_graph)
        self.model_id = model
        self.ner = NamedEntityRecognition(self.model_id)
        self.ned = NamedEntityDisambiguation(self.model_id, dense_index_name, sparse_index_name)

    def nerd(self, question: str, debug: bool = False, logging: bool = False):
        if logging == False:   
            entities = self.ner.ner(question, debug=debug, logging=logging)
            if debug:
                print(entities)
                print()
            uris = []
            for e in entities:
                prediction = self.ned.ned(question, e, self.endpoint, debug=debug, logging=logging)
                if prediction:
                    uris.append(prediction)
            return uris
        else:
            entities, ner_logs = self.ner.ner(question, debug=debug, logging=logging)
            ned_logs = []
            uris = []
            for e in entities:
                prediction, log = self.ned.ned(question, e, self.endpoint, debug=debug, logging=logging)
                ned_logs.append(log)
                if prediction:
                    uris.append(prediction)
            return list(set(uris)), { "ner": ner_logs, "ned": ned_logs }

    def get_name(self):
        return "dev-linker-" + self.model_id.value

    def supported_targets(self) -> List[KnowledgeGraph]:
        return [kg for kg in KnowledgeGraph]


if __name__ == '__main__':    
    entity_linker = DevLinker(KnowledgeGraph.BEASTIARY_KG, dense_index_name='beastiary-faiss', sparse_index_name='beastiary-bm25')
    entities = entity_linker.nerd(question="which creatures not speaking draconic language do have chaotic good alignment?", debug=True)
    print(entities)
    
    # entity_linker = DevLinker(KnowledgeGraph.STELAR_KG, index_name='stelar-faiss')
    # entities = entity_linker.nerd(question="What is the license of d4da009f-190c-4288-b0be-9d9b12e22fc1 published by UnitedStatesGeologicalSurvey?", debug=True)
    # print(entities)

    # entity_linker = DevLinker(KnowledgeGraph.ELECTIONS_KG, 'pnyqa-faiss')
    # entities = entity_linker.nerd(question="Which counties in California were won by the Republican party?", debug=False)
    # print(entities)
    
    # entity_linker = DevLinker(KnowledgeGraph.ELECTIONS_KG, index_name='terraq-filtered-faiss')
    # entities = entity_linker.nerd(question="Find 12 sentinel-2 images with cloud coverage over 15% and vegetation percentage less than 10%.", debug=True)
    # print(entities)
    
    # entity_linker = DevLinker(KnowledgeGraph.ELECTIONS_KG, index_name='terraq-filtered-faiss')
    # entities = entity_linker.nerd(question="Show me sentinel-1 images of the Helsinki subregion with vertical vertical polarisation.", debug=True)
    # print(entities)
    
    # entity_linker = DevLinker(KnowledgeGraph.YAGO2geo, dense_index_name='yago2geo-filtered-faiss')
    # entities = entity_linker.nerd(question="What is the size of the municipality of Athens?", debug=True)
    # print(entities)
    