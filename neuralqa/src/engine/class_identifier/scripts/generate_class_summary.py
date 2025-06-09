from src.datasets.elections_datasets import ELECTIONS_CLASSES
from src.engine.geoquestions1089 import YAGO2GEO_CLASSES

import openai
import torch
import re
from transformers import pipeline
from SPARQLWrapper import SPARQLWrapper, JSON
from random import sample


class ClassSummarizer():
    
    # TODO: I could also provide values for the predicates... for example 3 values per predicate...
    SUMMARIZATION_PROMPT = """
    You are an expert knowledge graph user. You are given an knowledge graph ontology class, a representative set of instances of this class, their common predicates.
    Your job is to describe the knowledge graph class in a few sentences. Strive for meanigful and specific descriptions of what the class is. I want you to give me meaningful insight.
    Avoid referring to the predicates of the class. Avoid being too abstract and general. Be specific, based on the given examples. Put your description inside {{ }}.
    
    Class: {kg_class}
    Instances: {instances}
    Common predicates: {common_predicates}
    
    Your response:
    """
    
    # Note: Telling the model to not explain reduces performance probably...
    # SUMMARIZATION_ENHANCEMENT_PROMPT = """
    # You are an expert knowledge graph user. You are given an knowledge graph ontology class, a representative set of instances of this class, their common predicates
    # and a set of descriptions. 
    # Your task is to improve the description of the given class to make it more distinct. Focus on what is unique about the class, compared to its counterparts. Avoid referring
    # to the other classes in your description. You can make your description as large as you want. Put the final description inside {{ }}.
    
    # Class: {kg_class}
    # Instances: {instances}
    # Common predicates: {common_predicates}
    # Descriptions: 
    # {descriptions}
    
    # Your response:
    # """
    
    # Note: Telling the model to not explain reduces performance probably...
    # SUMMARIZATION_ENHANCEMENT_PROMPT = """
    # You are an expert knowledge graph user. You are given an knowledge graph ontology class, a representative set of instances of this class, their common predicates
    # and a set of descriptions. 
    # Your task is to improve the description of the given class to make it more distinct. Focus on what is unique about the class, compared to its counterparts. Avoid referring
    # to the other classes in your description. You must make your description direct and to  the point, keeping only what is neccessary with a focus on uniqueness. 
    # Put the final description inside {{ }}.
    
    # Class: {kg_class}
    # Instances: {instances}
    # Common predicates: {common_predicates}
    # Descriptions: 
    # {descriptions}
    
    # Your response:
    # """
    
    SUMMARIZATION_ENHANCEMENT_PROMPT = """
    You are an expert knowledge graph user. You are given an knowledge graph ontology class, a representative set of instances of this class, their common predicates
    and a set of descriptions. 
    Your task is to extract keywords that best describe the class. Focus on what is unique about the class, how it differs from its counterparts. Avoid referring
    to the other classes in your keywords. Imagine that these keywords will be used for easy retrieval by users in a keyword-based search system. 
    Put the final set of keywords inside {{ }}.
    
    Class: {kg_class}
    Instances: {instances}
    Common predicates: {common_predicates}
    Descriptions: 
    {descriptions}
    
    Your response:
    """
    
    SPARQL_GET_CLASSES = """
    SELECT DISTINCT ?class
    WHERE {
        ?instance a ?class .
    }
    ORDER BY ?class
    """
    
    SPARQL_GET_INSTANCES = """
    SELECT ?x
    WHERE {{
        ?x a <{kg_class}>
    }}
    """
    # We limit the result set via sampling in the Python code.
    
    SPARQL_GET_COMMON_PREDICATES = """
    SELECT DISTINCT ?y
    WHERE {{
        # Count total instances
        {{
            SELECT (COUNT(DISTINCT ?x) AS ?total)
            WHERE {{
            ?x rdf:type <{kg_class}> .
            }}
        }}
        
        # Find predicates used by all instances
        {{
            SELECT ?y (COUNT(DISTINCT ?x) AS ?count)
            WHERE {{
            ?x rdf:type <{kg_class}> .
            ?x ?y ?z .
            }}
            GROUP BY ?y
        }}
        
        FILTER(?count = ?total)
    }}
    """
    
    
    def __init__(self, knowledge_graph: str, endpoint_url: str):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        self.endpoint_url = endpoint_url
        
        self.model_id = "google/gemma-3-12b-it"
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": True,
            },
            device_map="auto",
        )
        
        self.client = openai.OpenAI(
            api_key='YOUR_OPENAI_API_KEY',
        )
           
    def process_knowledge_graph(self, knowledge_graph):
        print(f"Processing knowledge graph: {knowledge_graph}")
        
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setCredentials("user", "PASSWORD")
        sparql.setReturnFormat(JSON)
        
        # classes = []
        # sparql.setQuery(ClassSummarizer.SPARQL_GET_CLASSES)
        # try:
        #     ret = sparql.queryAndConvert()

        #     for r in ret["results"]["bindings"]:
        #         c = r['class']['value']
        #         # print(c)
        #         classes.append(c)
        # except Exception as e:
        #     print(e)
        
        # print("SUMMARIZATION STAGE BEGINS")
        
        # classes = YAGO2GEO_CLASSES
        classes = ELECTIONS_CLASSES
        responses = []
        for kg_class in classes:
            print(kg_class)
            sparql.setQuery(ClassSummarizer.SPARQL_GET_INSTANCES.format(kg_class=kg_class))
            instances = []
            try:
                ret = sparql.queryAndConvert()
                instances = [r['x']['value'] for r in ret["results"]["bindings"]]
                if len(instances) > 10:
                    instances = sample(instances, 10)
                # print(instances)
            except Exception as e:
                print(e)
                
            sparql.setQuery(ClassSummarizer.SPARQL_GET_COMMON_PREDICATES.format(kg_class=kg_class))
            predicates = []
            try:
                ret = sparql.queryAndConvert()
                predicates = [r['y']['value'] for r in ret["results"]["bindings"]]
                # print(predicates)
            except Exception as e:
                print(e)
                
            messages = [
                {"role": "user", "content": ClassSummarizer.SUMMARIZATION_PROMPT.format(
                    kg_class=kg_class, instances=instances, common_predicates=predicates
                )},
            ]
            # print(messages)

            outputs = self.pipe(messages, max_new_tokens=512)
            assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
            
            # response = self.client.chat.completions.create(
            #     model="gpt-4o",
            #     # response_format={ "type": "json_object" },
            #     messages=messages
            # )
            # assistant_response = response.choices[0].message.content
            
            print(assistant_response)
            
            match = re.search(r"\{(.*?)\}", assistant_response)
            if match:
                assistant_response = match.group(1)
            
            responses.append((kg_class, assistant_response))
            
        with open(self.knowledge_graph + "_desc_k_12b.txt", "w+") as file:
            file.write('\n'.join(f"{x} - {y}" for x, y in responses))
            
        print("ENHANCEMENT STAGE BEGINS")
        
        responses_enhanced = []
        for response in responses:
            kg_class, assistant_response = response
            print(kg_class)
            
            sparql.setQuery(ClassSummarizer.SPARQL_GET_INSTANCES.format(kg_class=kg_class))
            instances = []
            try:
                ret = sparql.queryAndConvert()
                instances = [r['x']['value'] for r in ret["results"]["bindings"]]
                if len(instances) > 10:
                    instances = sample(instances, 10)
                # print(instances)
            except Exception as e:
                print(e)
                
            sparql.setQuery(ClassSummarizer.SPARQL_GET_COMMON_PREDICATES.format(kg_class=kg_class))
            predicates = []
            try:
                ret = sparql.queryAndConvert()
                predicates = [r['y']['value'] for r in ret["results"]["bindings"]]
                # print(predicates)
            except Exception as e:
                print(e)
                
            messages = [
                {"role": "user", "content": ClassSummarizer.SUMMARIZATION_ENHANCEMENT_PROMPT.format(
                    kg_class=kg_class, instances=instances, common_predicates=predicates, descriptions='\n'.join(f"{x}-{y}" for x, y in responses)
                )},
            ]
            # print(messages)
            
            # outputs = self.pipe(messages, max_new_tokens=512)
            # assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                # response_format={ "type": "json_object" },
                messages=messages
            )
            assistant_response = response.choices[0].message.content
            
            print(assistant_response)
            
            match = re.search(r"\{(.*?)\}", assistant_response)
            if match:
                assistant_response = match.group(1)
            
            responses_enhanced.append((kg_class, assistant_response))
        
        with open(self.knowledge_graph + "_desc++_k_12b.txt", "w+") as file:
            file.write('\n'.join(f"{x} - {y}" for x, y in responses_enhanced))


if __name__ == '__main__':
    summarizer = ClassSummarizer(knowledge_graph="elections", 
                                 endpoint_url="http://195.134.71.116:7200/repositories/pnyqa_kg_2")
    
    summarizer.process_knowledge_graph("elections")
