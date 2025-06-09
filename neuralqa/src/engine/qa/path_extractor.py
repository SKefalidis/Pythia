import time
import re
import traceback

from src.logging import log, LogComponent, LoggingOptions, LogType, Colors, LogLevel, create_logger
from src.datasets.dataset import KnowledgeGraph, uri_to_uril, uril_to_uri, triples_with_urils_to_triples_with_uris, triples_with_uris_to_triples_with_urils, urils_to_uris, uril_to_uri_map
from src.utils import SupportedLLMs, embed, execute_sparql_query, get_kgaqa_tracker, llm_call, is_entity_placeholder, is_property_description, is_type_predicate, is_uri
import jellyfish
import traceback


_PROMPT_PRELIMINARIES = """
A graph reasoning path is a sequence of connections between entities and classes in a Knowledge Graph that can be used to retrieve a subgraph of the Knowledge Graph relevant to a specific question.
The graph reasoning path essentially describes how to navigate through the graph to collect all the information that is required to answer the question.
A graph reasoning path always begins with a class or an entity and ends with a class, an entity, or a value.

These are some examples of reasoning paths:
Question: Sergios is the brother of Christina.
Context given: http://knowledge.com/resource/Sergios, http://knowledge.com/resource/Christina
Reasoning path: <http://knowledge.com/resource/Sergios> -> <http://knowledge.com/resource/Christina>. This path means get the paths that connect Sergios to Christina.
In this case the reasoning path consists only of entities. The graph path that best connects these entities will be discovered. 
Notice how I don't use the "brother" keyword. We only include entities and classes in the graph reasoning path. 
The "brother" keyword is a property that connects the two entities, so it is not correct to insert it.

Question: Who is the tallest person in the world?
Context given: http://knowledge.com/ontology/Person
Reasoning path: <http://knowledge.com/ontology/Person> -> "height". This path means get the property "height" for all the members of the class http://knowledge.com/ontology/Person.
The reasoning path starts with a class. Classes do not directly have a height. 
Implicitly, all members of the class will be collected, and then the the predicate that matches "height" the best will be used to rank them.
This reasoning path ends with a value, we describe the value, in this case the height of the person.
    
Essentially, a reasoning path is a sequence of connections between nodes of the Knowledge Graph that lead to the answer.

In this task, we have a knowledge graph and a question that we want to answer using the knowledge graph.
To do so, we first identify a reasoning path that leads to the answer, and then we ground that reasoning path in the knowledge graph.
"""

_PROMPT_TASK_EXPLANATION_GROUNDING = """
You are given a question and a reasoning path (a sequence of one-hop relations) that leads to the answer. Your reasoning path might be partial, meaning that it might be part of a bigger solution.
You will be given sets of triples that represent connections between entities and classes in the knowledge graph.
Your job is to select the top-3 best groundings for the reasoning path.

To answer select the number represeting the set of triples. You must put each one of your chosen numbers inside separate curly braces { }.
You must only put numbers inside curly braces, no other text. 
You must select at least one triple set and the order of your choices matters.
The first choice will be the first used, if that does not work out, the second one will be used, and so on.
"""

_PROMPT_TASK_EXPLANATION_DECIDE_INCLUSION = """
You are given a question and a reasoning path (a sequence of one-hop relations) that leads to the answer. Your reasoning path might be partial, meaning that it might be part of a bigger solution.
You will be given sets of triples that represent connections between entities and classes in the knowledge graph.
Your job is to decide if it is very likely that the given sets of triple adequately describe the connection in the context of the question.

Answer with either YES or NO inside curly braces {{ }}. If you are unsure, prefer to answer NO.

For example:
Question: Is Sergios the brother of Christina?
Reasoning path: <http://knowledge.com/resource/Sergios> -> <http://knowledge.com/resource/Christina>.
Triples given: 1. <http://knowledge.com/resource/Sergios> <http://knowledge.com/property/hasBusinessPartner> <http://knowledge.com/resource/Christina> .

This triple is not enough to answer the question. Even though there is a connection between the two entities, it does not say that they are siblings.

Another example:
Question: Is Lake Pamvotida in Greece?
Reasoning path: <http://knowledge.com/ontology/Pamvotida> -> <http://knowledge.com/ontology/Greece>.
Triples given: 1. <http://knowledge.com/ontology/Pamvotida> <http://knowledge.com/property/hasCountry> <http://knowledge.com/ontology/Greece> .
2. <http://knowledge.com/ontology/Pamvotida> <http://knowledge.com/property/page_link> <http://knowledge.com/ontology/Ioannina> .

The first triple is surely enough to answer the question. The second triple is ambiguous, and by itself would not be enough for us to be certain.
"""

_PROMPT_TASK_PROMPT_GROUNDING = """
For the relation ( {relation} ) of the reasoning path, select the top-3 most fitting sets of triples from the knowledge graph in the context of the question "{sentence}". 
Each triple set is accompanied by the number of nodes in the Knowledge Graph (I call that popularity). In other words the number of triples that it matches. Take that into account when selecting the triples, does it make sense for a triple to have many or a few matches? Think with your knowledge of the world, not just the triples.
There might also be some sample values for the triples, but they are not always present.
The triples are extracted from the knowledge graph, they are not guesses, so you can be certain that the connections exist in the knowledge graph.
You can use the popularity to break ties, but the most important factor is the meaning. If you know information about the question that is being asked, you can use it to reason about the triples.
"""

_PROMPT_TASK_PROMPT_DECIDE_INCLUSION = """
For the relation ( {relation} ) of the reasoning path, decide if it is very likely, not just a possibility, that the available sets of triples can adequately describe the connection in the context of the question "{sentence}". 
Each set of triples is accompanied by its popularity in the Knowledge Graph.

**Think step-by-step and explain your reasoning before answering.**
""" # we want this to think, because thinking saves us time here :^)

_GENERAL_PROMPT = """
{preliminaries}

{task_explanation}

Question: {sentence}
Reasoning path: {path}

{task_prompt}

Triples given:
{triples}

You must begin the final answer with the string # FINAL ANSWER.**

Answer:
""" 

_PROMPT_SELECT_PREDICATE_FOR_PROPERTY = """
Everything I say is in the context of trying to answer the question "{sentence}", using a knowledge graph. I need help with that.

I am searching for a property (or predicate, they mean the same) of the knowledge graph node "{nodeA}". The property should match the description {nodeB}, whatever that means in the context of the question.
I don't know exactly which property I am looking for, but I have a list of candidate properties that I think might match the description.
Try to understand the request made in the question "{sentence}", the role that the graph node "{nodeA}" plays in answering the question and what the property that I want really entails.
Along the list of candidate predicates/properties I also give you the popularity of each predicate, which is a measure of how often the predicate is used in the knowledge graph.

Your task is as follows:
1. Understand the request made in the question "{sentence}".
2. Understand the role that the graph node "{nodeA}" plays in answering the question.
3. Understand what the property that I want really entails, based on the description {nodeB} and the context of the question.

Now after that is done you must do one of the following three things:
a) Select the predicates that best match the description {nodeB} of the knowledge graph resource {nodeA}. In other words, if you think that the property that I'm looking for is one of the candidate predicates, 
select it. It might be that multiple properties match the description, so you can select multiple predicates. Knowledge Graphs are big, so it is not uncommon to have multiple properties that match the description.
But you must also rank them, from a number of 1 to 5, to show me the most relevant ones. 5 is the most relevant, 1 is the least relevant.
e.g. The question is "What is the height of the tallest person?", the graph node is "http://knowledge.com/ontology/Person" and the property description is "the height of the tallest person on Earth".
The candidate predicates are (predicate - popularity):
http://knowledge.com/property/height - 1000 (this is the popularity of the predicate)
http://knowledge.com/property/hasHeight - 500
http://knowledge.com/property/hasSize - 200
Both http://knowledge.com/property/height and http://knowledge.com/property/hasHeight match the description, and we can't be sure which one contains the height of the tallest person. So your output should be:

MATCHING PREDICATES DETECTED
`http://knowledge.com/property/height - 5`
`http://knowledge.com/property/hasHeight - 5`
`http://knowledge.com/property/hasSize - 1`

I put the same ranking for both properties, because I think they are equally relevant to the description. If one was lower than the other, I might have missed the tallest person.
If one predicate had multiple orders of magnitude more popularity than the other, I would have put it higher, but in this case they are close enough that I think they are equally relevant.
Do note that I put the answers in separate backticks, you must do the same.

That was the first possible case for your task, now the second one:
b) If you believe that none of the candidate predicates are good matches, it might be that the property that I'm searching for is not directly connected to {nodeA}. 
Instead we might first need to go to an intermediate entity, and through that reach our goal. In this case you must choose which predicates should be used to reach that intermediate entity.
Again, you must rank them from 1 to 5, where 5 is the most relevant, and 1 is the least relevant.
e.g. The question is "What kind of research does Sergios conduct for his PhD?", the graph node is "http://knowledge.com/resource/Sergios" and the property description is "the research topic of Sergios' PhD".
The candidate predicates are (predicate - popularity):
http://knowledge.com/property/hasHeight - 1 (again this is the popularity of the predicate)
http://knowledge.com/property/age - 1 
http://knowledge.com/property/occupation - 2
http://knowledge.com/property/pets - 3
In this case, none of the predicates match the description, but we can use the occupation predicate to reach an intermediate entity, and then from there we might be able to get the research topic. 
So your output should be:

INTERMEDIATE PREDICATES DETECTED
`http://knowledge.com/property/occupation - 4`

I completely ignore the other predicates, because they are not at all relevant to the description. We only rank predicates that could be even a little bit relevant. Occupation is decently relevant, so I put 4.

c) If you neither find a matching predicate, nor a predicate that could lead to an intermediate entity, you can select no predicates at all.
In this case, you must write NONE inside backticks in the answer. Use this option only if you are sure that no predicates match the description {nodeB} of the knowledge graph resource {nodeA}.

Some tips:
- Try to think what the popularity of each predicate entails. Use it to break ties, but don't use it as the only factor.
- Is {nodeA} a specific entity or a class? This might change the way you think about the predicates.
- Don't make up your own predicates, use the ones I give you.
- Please put only your answers inside backticks. Do not put any other text inside backticks, only the predicates that you select and rank. For your explanation don't use backticks, just write it normally.
- You must write one of MATCHING PREDICATES DETECTED, INTERMEDIATE PREDICATES DETECTED or NONE in the beginning of your answer, so I can understand your response.

Time to do it!

Question: {sentence}
Graph node: {nodeA}
Property description: {nodeB}
Candidate predicates:
{predicates_string}

Your answer:
"""


is_class_index = {}
is_entity_index = {}
path_to_triples_index = {}
ENDPOINT = ""
KNOWLEDGE_GRAPH = ""


from anytree import Node, RenderTree
from typing import List

class TreeNode:
    def __init__(self, predicate_from, values, popularity, kg):
        self.kg = kg
        self.predicate_from = predicate_from
        self.popularity = popularity
        self.values = values
        if isinstance(values, List):
            if is_uri(values[0]):
                for i in range(len(values)):
                    self.values[i] = uri_to_uril(values[i], kg)
                self.type = "URI"
            else:
                self.type = "LITERAL"
        else:
            if is_uri(values):
                self.type = "URI"
            else:
                self.type = "LITERAL"
            self.values = [values]
    
    def __repr__(self):
        if self.type == "URI":
            return f"{self.predicate_from} -> {', '.join(self.values)}"
        else:
            return f"{self.predicate_from} -> {', '.join(self.values)}"
    
    def values_to_string(self):
        if self.type == "URI":
            urils = [uri_to_uril(value, self.kg) for value in self.values]
            return ", ".join([f"<{value}>" for value in urils])
        else:
            return ", ".join([f'"{value}"' for value in self.values])
                
def get_leafs(node: Node):
    leafs = []
    if node.is_leaf:
        leafs.append(node)
    else:
        for child in node.children:
            leafs += get_leafs(child)
    return leafs


class PathExtractor():

    def __init__(self, knowledge_graph: KnowledgeGraph, model_id_main: SupportedLLMs = SupportedLLMs.VLLM, model_id_explore: SupportedLLMs = SupportedLLMs.VLLM) -> None:
        super().__init__()
        self.model_id_main = model_id_main
        self.model_id_explore = model_id_explore
        self.knowledge_graph = knowledge_graph
        self.endpoint = KnowledgeGraph.get_endpoint(self.knowledge_graph)
        global ENDPOINT
        ENDPOINT = self.endpoint
        global KNOWLEDGE_GRAPH
        KNOWLEDGE_GRAPH = self.knowledge_graph
        self.ontology_endpoint = KnowledgeGraph.get_ontology_endpoint(self.knowledge_graph)
        self.tracker = get_kgaqa_tracker()
        
    # --------------------------
    # ----- SPARQL Queries -----
    # --------------------------
    
    def is_class(self, node: str):
        if not isinstance(node, str):
            return False
        og_node = node
        if node not in is_class_index:
            node = uril_to_uri(node)
            if self.knowledge_graph == KnowledgeGraph.WIKIDATA:
                query = f"""
                ASK WHERE {{ 
                    FILTER EXISTS {{ ?s <http://www.wikidata.org/prop/direct/P31> {node} }}
                }}
                """
            else:
                query = f"""
                ASK WHERE {{ 
                    FILTER EXISTS {{ ?s a {node} }}
                }}
                """
            # print(query)
            try:
                query_result = execute_sparql_query(query, self.endpoint)
                is_class_index[og_node] = query_result.convert()['boolean']
            except Exception as e:
                log(f"Error is_class: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                is_class_index[og_node] = False
        return is_class_index[og_node]
    
    def is_entity(self, node: str):
        if not isinstance(node, str):
            return True
        og_node = node
        if node not in is_entity_index:
            if self.is_class(node):
                return False
            node = uril_to_uri(node)
            query = f"""
            ASK WHERE {{ 
                {{
                    FILTER EXISTS {{ ?s ?p {node} }}
                }}
                UNION
                {{
                    FILTER EXISTS {{ {node} ?p ?o }}
                }}
            }}
            """
            # print(query)
            try:
                query_result = execute_sparql_query(query, self.endpoint)
                is_entity_index[og_node] = query_result.convert()['boolean']
            except Exception as e:
                log(f"Error is_entity: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                is_entity_index[og_node] = False
        return is_entity_index[og_node]
    
    def get_types_for_node(self, node: str):
        node = uril_to_uri(node)
        if self.knowledge_graph == KnowledgeGraph.WIKIDATA:
            query = f"""
            SELECT ?type
            WHERE {{
                {node} <http://www.wikidata.org/prop/direct/P31> ?type .
            }}
            """
        else:
            query = f"""
            SELECT ?type
            WHERE {{
                {node} a ?type .
            }}
            """
        # print(query)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_types_for_node: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return []
        
        types = []
        for result in results["results"]["bindings"]:
            types.append(result["type"]["value"])
        types = [uri_to_uril(type, self.knowledge_graph) for type in types]
        return types
    
    def get_get_all_paths_from_to(self, from_node: str, to_node: str, depth_limit: int, endpoint: str, bidirectional: str = "true"):
        self.tracker._pe_all_paths_calls += 1
        start_time = time.time()
        
        from_node = uril_to_uri(from_node)
        to_node = uril_to_uri(to_node)
        query = f"""
        PREFIX path: <http://www.ontotext.com/path#>

        SELECT ?predicatePath (COUNT(?predicatePath) AS ?pathCount)
        WHERE {{
        {{
            SELECT ?pathIndex (GROUP_CONCAT(STR(?p); SEPARATOR=" -> ") AS ?predicatePath)
            WHERE {{
            {{
                SELECT ?pathIndex ?p
                WHERE {{
                VALUES (?src ?dst) {{
                    ({from_node} {to_node})
                }}

                SERVICE path:search {{
                    [] path:findPath path:allPaths ;
                    path:sourceNode ?src ;
                    path:destinationNode ?dst ;
                    path:pathIndex ?pathIndex ;
                    path:resultBindingIndex ?edgeIndex ;
                    path:resultBinding ?edge ;
                    path:propertyBinding ?p ;
                    path:maxPathLength {depth_limit} ;
                    path:bidirectional {bidirectional} .
                }}
                }}
                ORDER BY ?pathIndex ?edgeIndex
            }}
            }}
            GROUP BY ?pathIndex
        }}
        }} GROUP BY ?predicatePath
        """
        # print(query)
        
        try:
            query_result = execute_sparql_query(query, endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_get_all_paths_from_to (probably ran out of time): {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            self.tracker._pe_all_paths_time += time.time() - start_time
            return [], []
        # print(results)
        
        paths, popularity = [], []
        for result in results["results"]["bindings"]:
            paths.append(result["predicatePath"]["value"])
            popularity.append(int(result["pathCount"]["value"]))
        # paths = [uri_to_uril(path, self.knowledge_graph) for path in paths]
        log(f"Found {len(paths)} paths with get_get_all_paths_from_to...", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        for j in range(len(paths)):
            uris = paths[j].split(" -> ")
            for i in range(len(uris)):
                uris[i] = uri_to_uril(uris[i], self.knowledge_graph)
            paths[j] = " -> ".join(uris)
        self.tracker._pe_all_paths_time += time.time() - start_time
        return paths, popularity
    
    def get_shortest_path_from_to(self, from_node: str, to_node: str, endpoint: str, bidirectional: str = "true"):
        self.tracker._pe_shortest_calls += 1
        start_time = time.time()
        
        from_node = uril_to_uri(from_node)
        to_node = uril_to_uri(to_node)
        query = f"""
        PREFIX path: <http://www.ontotext.com/path#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?predicatePath (COUNT(?predicatePath) AS ?pathCount)
        WHERE {{
          {{
            SELECT ?pathIndex (GROUP_CONCAT(STR(?p); SEPARATOR=" -> ") AS ?predicatePath)
            WHERE {{
              {{
                SELECT ?pathIndex ?p
                WHERE {{
                  VALUES (?src ?dst) {{
                    ({from_node} {to_node})
                  }}
        
                  SERVICE path:search {{
                    [] path:findPath path:shortestPath ;
                       path:sourceNode ?src ;
                       path:destinationNode ?dst ;
                       path:pathIndex ?pathIndex ;
                       path:resultBindingIndex ?edgeIndex ;
                       path:resultBinding ?edge ;
                       path:propertyBinding ?p ;
                       path:bidirectional {bidirectional}.
                  }}
                }}
                ORDER BY ?pathIndex ?edgeIndex
              }}
            }}
            GROUP BY ?pathIndex
          }}
        }} GROUP BY ?predicatePath
        """
        # print(query)
        
        try:
            query_result = execute_sparql_query(query, endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_shortest_path_from_to (ran out of time, but this is important): {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            self.tracker._pe_shortest_path_time += time.time() - start_time
            return [], []
        # print(results)
        
        paths, popularity = [], []
        for result in results["results"]["bindings"]:
            paths.append((result["predicatePath"]["value"]))
            popularity.append(int(result["pathCount"]["value"]))
        log(f"Found {len(paths)} paths with get_shortest_path_from_to", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        for j in range(len(paths)):
            uris = paths[j].split(" -> ")
            for i in range(len(uris)):
                uris[i] = uri_to_uril(uris[i], self.knowledge_graph)
            paths[j] = " -> ".join(uris)
            # print(paths[j])
        self.tracker._pe_shortest_path_time += time.time() - start_time
        return paths, popularity
    
    def get_distinct_predicates_for_class(self, node: str, debug=False, filter_literals=False):
        node = uril_to_uri(node)
        if self.knowledge_graph != KnowledgeGraph.WIKIDATA:
            if not filter_literals:
                query = f"""
                SELECT ?p (COUNT(*) AS ?count)
                WHERE {{
                    {{
                        ?s a {node} .
                        ?s ?p ?o .
                    }}
                    UNION
                    {{
                        ?s a {node} .
                        ?x ?p ?s .
                    }}
                }}
                GROUP BY ?p
                ORDER BY DESC(?count)
                """
            else:
                query = f"""
                SELECT ?p (COUNT(*) AS ?count)
                WHERE {{
                    {{
                        ?s a {node} .
                        ?s ?p ?o .
                        FILTER (!isLiteral(?o))
                    }}
                    UNION
                    {{
                        ?s a {node} .
                        ?x ?p ?s .
                    }}
                }}
                GROUP BY ?p
                ORDER BY DESC(?count)
                """
        else:
            query = f"""
            SELECT ?p (COUNT(*) AS ?count)
            WHERE {{
                {{
                    ?s <http://www.wikidata.org/prop/direct/P31> {node} .
                    ?s ?p ?o .
                }}
                UNION
                {{
                    ?s <http://www.wikidata.org/prop/direct/P31> {node} .
                    ?x ?p ?s .
                }}
            }}
            GROUP BY ?p
            ORDER BY DESC(?count)
            """
            
        log(f"distinct predicates for class {node}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_distinct_predicates_for_class: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            return [], []
        
        predicates, popularity = [], []
        for result in results["results"]["bindings"]:
            predicates.append(result["p"]["value"])
            popularity.append(int(result["count"]["value"]))
        predicates = [uri_to_uril(predicate, self.knowledge_graph) for predicate in predicates]
        return predicates, popularity
    
    def get_distinct_predicates_for_entity(self, node: str, debug=False, filter_literals=False):
        node = uril_to_uri(node)
        if not filter_literals:
            query = f"""
            SELECT ?p (COUNT(*) AS ?count)
            WHERE {{
                {{
                    {node} ?p ?o .
                }}
                UNION
                {{
                    ?s ?p {node} .
                }}
            }}
            GROUP BY ?p
            ORDER BY DESC(?count)
            """
        else:
            query = f"""
            SELECT ?p (COUNT(*) AS ?count)
            WHERE {{
                {{
                    {node} ?p ?o .
                    FILTER (!isLiteral(?o))
                }}
                UNION
                {{
                    ?s ?p {node} .
                }}
            }}
            GROUP BY ?p
            ORDER BY DESC(?count)
            """
            
        log(f"distinct predicates for entity {node}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
        log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_distinct_predicates_for_entity: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            return [], []
        
        predicates, popularity = [], []
        for result in results["results"]["bindings"]:
            predicates.append(result["p"]["value"])
            popularity.append(int(result["count"]["value"]))
        # print(predicates)
        predicates = [uri_to_uril(predicate, self.knowledge_graph) for predicate in predicates]
        return predicates, popularity
    
    def get_object_for_subject_predicate(self, subject: str, predicate: str, limit=1, debug=False, cls=False):
        subject = uril_to_uri(subject)
        predicate = uril_to_uri(predicate)
        if cls == False:
            query = f"""
            SELECT ?o
            WHERE {{
                {subject} <{predicate}> ?o .
            }}
            LIMIT {limit}
            """
        else:
            if self.knowledge_graph != KnowledgeGraph.WIKIDATA:
                query = f"""
                SELECT ?o
                WHERE {{
                    ?var a {subject} .
                    ?var <{predicate}> ?o .
                }}
                LIMIT {limit}
                """
            else:
                query = f"""
                SELECT ?o
                WHERE {{
                    ?var <http://www.wikidata.org/prop/direct/P31> {subject} .
                    ?var <{predicate}> ?o .
                }}
                LIMIT {limit}
                """
        if debug:
            print(query)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_object_for_subject_predicate: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            return []
        
        objects = []
        for result in results["results"]["bindings"]:
            objects.append(result["o"]["value"])
        objects = [uri_to_uril(object, self.knowledge_graph) for object in objects]
        return objects
    
    def get_subject_from_predicate_object(self, predicate: str, object: str, limit=1, debug=False, cls=False):
        object = uril_to_uri(object)
        predicate = uril_to_uri(predicate)
        if cls == False:
            query = f"""
            SELECT ?s
            WHERE {{
                ?s <{predicate}> {object} .
            }}
            LIMIT {limit}
            """
        else:
            if self.knowledge_graph != KnowledgeGraph.WIKIDATA:
                query = f"""
                SELECT ?s
                WHERE {{
                    ?var a {object} .
                    ?s <{predicate}> ?var .
                }}
                LIMIT {limit}
                """
            else:
                query = f"""
                SELECT ?s
                WHERE {{
                    ?var <http://www.wikidata.org/prop/direct/P31> {object} .
                    ?s <{predicate}> ?var .
                }}
                LIMIT {limit}
                """
        if debug:
            print(query)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error get_subject_from_predicate_object: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            return []
        
        objects = []
        for result in results["results"]["bindings"]:
            objects.append(result["s"]["value"])
        objects = [uri_to_uril(object, self.knowledge_graph) for object in objects]
        return objects
    
    # --------------------------------------------------------
    # ----- Ground Reasoning Path to the Knowledge Graph -----
    # --------------------------------------------------------
    
    # def get_top_predicates(self, targets: List[str], predicates, top_k=50, include_most_popular=0):
    #     start = time.time()
    #     predicate_embeddings = [
    #         (predicate, embed(predicate.split('/')[-1]))
    #         for predicate in predicates
    #     ]
    #     get_kgaqa_tracker()._embed_time += time.time() - start
        
    #     top_predicates = []
        
    #     choices_per_target = (top_k // len(targets)) - include_most_popular
    #     for target in targets:
    #         # Jaro-Winkler sort
    #         sorted_jw = sorted(
    #             predicates,
    #             key=lambda s: jellyfish.jaro_winkler_similarity(s.split('/')[-1], target),
    #             reverse=True
    #         )
    #         top_jw = sorted_jw[:choices_per_target//2]

    #         # Cosine similarity sort
    #         print(f"Count of predicates: {len(predicates)}")
    #         nodeB_vec = embed(target)
    #         sorted_cos = sorted(
    #             predicate_embeddings,
    #             key=lambda item: cos_sim(item[1], nodeB_vec).item(),
    #             reverse=True
    #         )
    #         top_cos = [item[0] for item in sorted_cos[:choices_per_target//2]]
            
    #         top_predicates += top_jw + top_cos

    #     top_predicates += predicates[:include_most_popular]  # Include most popular predicates if specified
        
    #     return list(set(top_predicates))
    
    def get_paths_by_graph_search(self, sentence: str, reasoning_path: List[str], start: str, end: str, endpoint: str):
        get_kgaqa_tracker()._pe_graph_search_calls += 1
        start_time = time.time()
        
        shortest_paths, shortest_paths_popularity = self.get_shortest_path_from_to(start, end, endpoint)
        if shortest_paths == []:
            get_kgaqa_tracker()._pe_graph_search_time += time.time() - start_time
            return [], []
        shortest_path_length = len(shortest_paths[0].split(" -> "))
        if shortest_path_length < 5 and len(shortest_paths) < 10:
            triples_strings = []
            triples_popularities = []
            for property_path, popularity in zip(shortest_paths, shortest_paths_popularity):
                # print_colored("Property Path: " + property_path, Colors.BOLD)
                log(f"Property Path: {property_path}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                final_triples = self.property_path_to_triples(start, end, property_path)
                if final_triples == []:
                    log(f"Empty triples for property path: {property_path}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                    continue
                
                for idx, triples in enumerate(final_triples):
                    triples_strings.append(triples)
                    triples_popularities.append(popularity)
                    log(f"\t{idx}. {triples}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            enumerated_triples_string = "\n".join(f"{idx}. popularity: {popularity} triples: {triple_string}" for idx, (triple_string, popularity) in enumerate(zip(triples_strings, triples_popularities)))
            
            # ask LLM for an assesment
            prompt = _GENERAL_PROMPT.format(
                preliminaries=_PROMPT_PRELIMINARIES,
                task_explanation=_PROMPT_TASK_EXPLANATION_DECIDE_INCLUSION,
                sentence=sentence,
                path=" ".join(reasoning_path),
                task_prompt=_PROMPT_TASK_PROMPT_DECIDE_INCLUSION.format(relation=(start + " -> " + end), sentence=sentence),
                triples=enumerated_triples_string
            )
            log(f"Prompt for decide inclusion: {prompt}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.PROMPT)
        
            get_kgaqa_tracker()._pe_prompt_inclusion_calls += 1
            start_time2 = time.time()
        
            generated = llm_call(self.model_id_main, prompt, 1024)
            log(f"LLM response for decide inclusion: {generated}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.LLM_RESULT)
            
            get_kgaqa_tracker()._pe_prompt_inclusion_time += time.time() - start_time2
            
            if "FINAL ANSWER" in generated:
                generated = generated.split("FINAL ANSWER")[-1]
            
            if r"{NO}" in generated:
                log("Search for more paths (with all paths on GraphDB)", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                paths, paths_popularity = self.get_get_all_paths_from_to(start, end, shortest_path_length + 1, endpoint) # search for paths with one more hop
                if paths == [] or len(paths) > 100: # pruning too many paths for performance, without this it would probably get better results, but it would take too long
                    paths = shortest_paths
                    paths_popularity = shortest_paths_popularity
            else:
                paths = shortest_paths
                paths_popularity = shortest_paths_popularity
        else:
            paths = shortest_paths
            paths_popularity = shortest_paths_popularity
        get_kgaqa_tracker()._pe_graph_search_time += time.time() - start_time
        return paths, paths_popularity
    
    def get_paths_by_neighborhood_search(self, question: str, start, target: str, initial_depth=2, to_uri=False):
        get_kgaqa_tracker()._pe_neighborhood_calls += 1
        start_time = time.time()
        
        total_graph = 0
        total_llm = 0
        
        if isinstance(start, str):
            root = Node(TreeNode("root", [start], 1, self.knowledge_graph))
            is_class = self.is_class(start)
            log(f"get_paths_by_neighborhood_search: start is a string: {start}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
        elif isinstance(start, list):
            root = Node(TreeNode("root", start, 1, self.knowledge_graph))
            is_class = False
            log(f"get_paths_by_neighborhood_search: start is a list of strings: {start}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
        else:
            raise ValueError("Start must be a string or a list of strings.")

        depth = initial_depth
        while depth > 0:
            leafs = get_leafs(root)
            # print(f"The depth is: {depth}")
            for leaf in leafs:
                tree_node = leaf.name
                if tree_node.type == "LITERAL":
                    continue
                if tree_node.type == "URI":
                    for value in tree_node.values:
                        if value in uril_to_uri_map:
                            value = uri_to_uril(value, self.knowledge_graph)
                        value_graph = uril_to_uri(value)
                        value_readable = value
                        if value_graph[0] != "<" or value_graph[-1] != ">":
                            value_graph = "<" + value_graph + ">"
                        if (self.is_class(value_graph)) and not is_class:
                            is_class = False
                            continue

                        # For the values URIs get predicates and their popularity
                        start_time = time.time()
                        predicates_and_popularity = self.get_predicates_and_popularity_for_node(value_graph, filter_literals=to_uri)
                        if len(predicates_and_popularity) == 0:
                            log(f"No predicates found for {value_readable}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                            continue
                        predicates, popularity = predicates_and_popularity
                        
                        for p in predicates:
                            log(f"Predicate: {p} - Popularity: {popularity[p]}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                        total_graph += time.time() - start_time
                        
                        # avoid reusing the same paths
                        filtered_predicates = []
                        for p in predicates:
                            if tree_node.predicate_from == p:
                                continue
                            ancestors = leaf.ancestors
                            if any(ancestor.name.predicate_from == p for ancestor in ancestors):
                                continue
                            filtered_predicates.append(p)
                        if len(filtered_predicates) == 0:
                            log(f"No predicates left after filtering for {value_readable}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                            continue                       
                        predicates = filtered_predicates
                        
                        # this will be used to select the most relevant predicates
                        predicates_popularity = "\n".join([f"{p} - {popularity[p]}" for p in predicates])
                        
                        prompt = f"""
                        You are exploring a knowledge graph to find a connection between two concepts in the context of the question: "{question}".

                        - Initial node: "{start}"
                        - Current node: "{value_readable}"
                        - Goal: "{target}"
                        
                        Your task is to select the **three most relevant predicates** from the list below that are most likely to lead from the start node toward the goal concept. These predicates represent the known relationships starting from "{value_readable}", but they do not yet have expanded values.

                        The structure below shows the current node and its predicates, along with how popular each predicate is in the graph (higher count = more commonly used).

                        Choose the 3 predicates that seem most promising for discovering information about the goal: **"{target}"**

                        Predicates - Popularity:
                        {predicates_popularity}

                        Respond with only the URIs of the 3 selected predicates, one per line.
                        Do not include explanations or extra text.
                        
                        Answer:
                        """
                        log(f"Prompt for selecting predicates: {prompt}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.PROMPT)

                        # Call the LLM
                        start_time = time.time()
                        get_kgaqa_tracker()._pe_prompt_neighborhood_calls += 1
                        generated = llm_call(self.model_id_explore, prompt, max_tokens=512)
                        log(f"LLM response for selecting predicates: {generated}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.LLM_RESULT)
                        get_kgaqa_tracker()._pe_prompt_neighborhood_time += time.time() - start_time
                        total_llm += time.time() - start_time

                        selected_predicate_ids = [line.strip() for line in generated.strip().splitlines() if line.strip()]
                        
                        log(f"LLM selected predicates: {selected_predicate_ids}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
                        for predicate_id in selected_predicate_ids:
                            log(f"Selected predicate: {predicate_id}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                    
                        # get values for the selected predicates
                        for predicate_id in selected_predicate_ids:                        
                            log(f"Getting values for predicate {predicate_id}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
                            
                            # Get values for the predicate (both object and subject direction)
                            values = self.get_object_for_subject_predicate(value_graph, predicate_id, limit=3, cls=is_class, debug=False)
                            values += self.get_subject_from_predicate_object(predicate_id, value_graph, limit=3, cls=is_class, debug=False)

                            if len(values) == 0:
                                log(f"No values found for predicate {predicate_id}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
                                continue
                            if isinstance(start, str) and start in values[0]:
                                continue
                            elif isinstance(start, list) and any(s in values[0] for s in start):
                                continue
                            
                            values = urils_to_uris(values)
                            
                            # Create a new tree node for the values
                            if predicate_id in popularity:
                                new_tree_node = TreeNode(predicate_id, values, popularity[predicate_id], KNOWLEDGE_GRAPH)
                            else:
                                new_tree_node = TreeNode(predicate_id, values, 0, KNOWLEDGE_GRAPH)
                            new_node = Node(new_tree_node, parent=leaf)
                        
                        for pre, _, node in RenderTree(root):
                            log(f"{pre}{node.name}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                        
            depth -= 1
            is_class = False
            
        paths = []
        popularity_hashtable = {}
        for node in root.descendants:
            # print(node.name)
            ancestors = node.ancestors
            if len(ancestors) == 0:
                continue
            path = [ancestor.name.predicate_from for ancestor in ancestors if ancestor.name.type == "URI"][1:] + [node.name.predicate_from]
            if node.name.type == "LITERAL":
                path.append("PROPERTY")
            if len(path) > 0:
                paths.append(" -> ".join(path))
                # popularity_hashtable[" -> ".join(path)] = node.name.values_to_string() # leaf.name.popularity
                key = " -> ".join(path)
                value = node.name.values_to_string()
                if key in popularity_hashtable and value not in popularity_hashtable[key]:
                    popularity_hashtable[key] += f", {value}"
                else:
                    popularity_hashtable[key] = f"{node.name.popularity} | Sample Values: {value}"
                    
        paths = list(set(paths))  # Remove duplicates
        
        popularities = [popularity_hashtable[path] for path in paths]          
        
        log(f"Paths found: {paths}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
        for path, popularity in zip(paths, popularities):
            log(f"{path} - {popularity}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            
        log(f"Time taken to get predicates and popularity: {total_graph:.2f} seconds", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        log(f"Time taken to get selected predicates: {total_llm:.2f} seconds", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        get_kgaqa_tracker()._pe_neighborhood_time += time.time() - start_time
        
        return paths, popularities

    
    def get_candidate_paths_between_known_nodes(self, sentence: str, reasoning_path: List[str], nodeA: str, nodeB: str):
        #
        # setup search
        #
        if self.is_entity(nodeA) and self.is_entity(nodeB):
            log(f"Both nodes are entities: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            start = nodeA
            end = nodeB
            endpoint = self.endpoint
        elif self.is_class(nodeA) and self.is_entity(nodeB):
            log(f"Node A is a class, Node B is an entity: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            start = nodeA
            end = nodeB
            endpoint = self.endpoint
        elif self.is_entity(nodeA) and self.is_class(nodeB):
            log(f"Node A is an entity, Node B is a class: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            start = nodeA
            end = nodeB
            endpoint = self.endpoint
        elif self.is_class(nodeA) and self.is_class(nodeB):
            log(f"Both nodes are classes: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            start = nodeA
            end = nodeB
            endpoint = self.ontology_endpoint
        else:
            log(f"Invalid node types: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            log(f"Invalid node types: {uril_to_uri(nodeA)}, {uril_to_uri(nodeB)}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            traceback.print_stack()
            start = nodeA
            end = nodeB
            endpoint = self.endpoint
        #
        # graph search
        #
        paths, paths_popularity = self.get_paths_by_graph_search(sentence, reasoning_path, start, end, endpoint)
        #
        # post-process paths 
        #
        if self.is_class(nodeA) and self.is_class(nodeB):
            # Convert from ontology paths to property paths (the target KG is not the ontology)
            type_func = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            if self.knowledge_graph == KnowledgeGraph.WIKIDATA:
                type_func = "http://www.wikidata.org/prop/direct/P31"
            type_func = uri_to_uril(type_func, self.knowledge_graph)
            candidate_paths = []
            for path in paths:
                candidate_paths.append(type_func + " -> " + path + " -> " + type_func)
            candidate_paths_popularity = paths_popularity
        else:
            candidate_paths = paths
            candidate_paths_popularity = paths_popularity
        return candidate_paths, candidate_paths_popularity
    
    def get_predicates_and_popularity_for_node(self, node: str, filter_literals=False, debug=False):
        if self.is_entity(node):
            predicates, popularity = self.get_distinct_predicates_for_entity(node, filter_literals=filter_literals, debug=debug)
        elif self.is_class(node):
            predicates, popularity = self.get_distinct_predicates_for_class(node, filter_literals=filter_literals, debug=debug)
        else:
            log(f"Node {node} is neither an entity nor a class", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            return [], {}
        popularity_hashtable = dict(zip(predicates, popularity))
        return predicates, popularity_hashtable
    
    def get_predicates_and_popularity_for_nodes(self, nodes: List[str], debug=False):
        total_predicates = set()
        popularity_hashtable = {}
        for node in nodes:
            if not is_uri(node):
                continue
            if node[0] != "<" or node[-1] != ">":
                node = "<" + node + ">"
            if self.is_entity(node):
                predicates, popularity = self.get_distinct_predicates_for_entity(node)
            elif self.is_class(node):
                predicates, popularity = self.get_distinct_predicates_for_class(node)
            if not predicates:
                log(f"No predicates found for {node}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
                continue
            total_predicates.update(predicates)
            popularity_hashtable.update(dict(zip(predicates, popularity)))
        return total_predicates, popularity_hashtable
    
    # def get_synonyms_for_predicate(self, sentence: str, predicate: str, node: str):
    #     query = f"""In the context of the question "{sentence}", which must be answered using a knowledge graph:
    #     I am searching for a property {predicate} of the knowledge graph resource {node}.
    #     Can you suggest some synonyms for this property? Understand its meaning in the context of the question.
    #     If there are any synonyms, that are different enough to help me find the property, please provide them.
    #     Put your answers (max 3 most diverse synonyms) in separate curly braces {{ }}
    #     """
    #     generated = llm_call(self.model_id, query, 4096, 0.5)
    #     pat = r'(?<=\{).+?(?=\})'
    #     synonyms = re.findall(pat, generated)
    #     print_colored(f"Synonyms for predicate {predicate} in context of {sentence}: {synonyms}", Colors.YELLOW)
    #     return synonyms
        
    def get_candidate_paths(self, sentence: str, reasoning_path: List[str], nodeA: str, nodeB: str, placeholder: bool = False):
        candidate_paths = []
        candidate_paths_popularity = []
        log(f"Getting candidate paths for {nodeA} -> {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO, LogType.HEADER)
        
        if isinstance(nodeA, list):
            nodeA = nodeA[0]  # Take the first node if it's a list, skibidi
        
        if is_uri(nodeA) and is_uri(nodeB):
            candidate_paths, candidate_paths_popularity = self.get_candidate_paths_between_known_nodes(sentence, reasoning_path, nodeA, nodeB)
        elif is_uri(nodeA) and not is_uri(nodeB) and is_property_description(nodeB):
            log(f"Node A is a URI, Node B is a property description: {nodeA}, {nodeB}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            
            paths, popularity = self.get_paths_by_neighborhood_search(sentence, nodeA, nodeB)
            
            if paths == []:
                log(f"No predicates selected for {nodeA} -> {nodeB}, skipping...", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                return [], []
            
            for i in range(len(paths)):
                predicate_path = paths[i]
                candidate_path = predicate_path
                if "PROPERTY" not in candidate_path:
                    candidate_path += " -> PROPERTY"
                if self.is_entity(nodeA):
                    candidate_paths.append(candidate_path)
                    candidate_paths_popularity.append(popularity[i])
                elif self.is_class(nodeA):
                    type_func = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                    if self.knowledge_graph == KnowledgeGraph.WIKIDATA:
                        type_func = "http://www.wikidata.org/prop/direct/P31"
                    type_func = uri_to_uril(type_func, self.knowledge_graph)
                    candidate_paths.append(f"{type_func} -> " + candidate_path)
                    candidate_paths_popularity.append(popularity[i])
                    
        return candidate_paths, candidate_paths_popularity
        
    # -------------------------------------
    # ----- Property paths to triples -----
    # -------------------------------------
    
    def are_triples_valid(self, triples: str):
        triples = triples_with_urils_to_triples_with_uris(triples, self.knowledge_graph)
        
        query = f"""
        ASK WHERE {{ 
            {triples}
        }}
        """
        log(f"are_triples_valid: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            result = query_result.convert()
            return result['boolean']
        except Exception as e:
            log(f"Error are_triples_valid: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            # raise ValueError("Error are_triples_valid")
            return False
    
    def has_no_value_connection(self, triples: str):    
        triples = triples_with_urils_to_triples_with_uris(triples, self.knowledge_graph)
          
        query = f"""
        SELECT * WHERE {{ 
            {triples}
        }} LIMIT 1
        """
        log(f"has_no_value_connection: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error has_no_value_connection: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            return False
        keep = True
        for result in results["results"]["bindings"]:   
            for var in result:
                if (result[var]["type"]) == 'literal':
                    keep = False
                    break
            if keep == False:
                break
        return keep
    
    def replace_types_for_triples(self, triples: str):
        triples = triples_with_urils_to_triples_with_uris(triples, self.knowledge_graph)
        
        # find all variables that are types
        variables = []
        lines = triples.split("\n")
        for l in lines:
            if len(l) == 0:
                continue
            s, p, o = l.split(" ")[0], l.split(" ")[1], l.split(" ")[2]
            if is_type_predicate(p):
                if not o.startswith("<"):
                    variables.append(o)
        if len(variables) == 0:
            triples = triples_with_uris_to_triples_with_urils(triples, self.knowledge_graph)
            return triples
        variables = list(set(variables))
        
        query = f"""
        SELECT {" ".join(variables)} WHERE {{ 
            {triples}
        }} LIMIT 1
        """
        log(f"replace_types_for_triples: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        # execute query
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error replace_types_for_triples: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            triples = triples_with_uris_to_triples_with_urils(triples, self.knowledge_graph)
            return triples
        
        # replace variables with actual URIs
        for result in results["results"]["bindings"]:
            for v in variables:
                if v[1:] in result:
                    triples = triples.replace(v, "<" + result[v[1:]]["value"] + ">")
        triples = triples_with_uris_to_triples_with_urils(triples, self.knowledge_graph)
        return triples
        
    def property_path_to_triples(self, start: str, goal: str, path: str):
        key = f"{start}->{goal}->{path}"
        
        if key not in path_to_triples_index:
            get_kgaqa_tracker()._pe_property_path_to_triples_calls += 1
            start_time = time.time()
            
            #
            # Generate triples from the path
            #
            current = start # "<" + start + ">"
            triples = [""]
            var_index = 0
            properties = path.split(" -> ")
            check_value_connection = True
            for idx, p in enumerate(properties):
                if p == "PROPERTY": # this means that we are searching for a literal, not a node
                    check_value_connection = False
                    break
                if p == "PLACEHOLDER":
                    break
                new_triples = []
                if idx == len(properties) - 1:
                    new_var = goal # "<" + goal + ">"
                else:
                    new_var = "?var" + str(var_index)
                    var_index += 1
                for t in triples:
                    if "VALUES" not in current:
                        new_t_1 = t + current + " <" + p + "> " + new_var + " . \n"
                        new_t_2 = t + new_var + " <" + p + "> " + current + " . \n"
                        new_triples.append(new_t_1)
                        new_triples.append(new_t_2)
                    else:
                        new_t_1 = t + current + "?vals <" + p + "> " + new_var + " . \n"
                        new_t_2 = t + current + new_var + " <" + p + ">  ?vals" + " . \n"
                        new_triples.append(new_t_1)
                        new_triples.append(new_t_2)
                current = new_var
                triples = new_triples
                
            log(f"Count of candidate triples: {len(triples)}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            
            #
            # Only keep valid collections of triples, i.e., those that have results in the KG
            #
            valid_triples = [triple for triple in triples if self.are_triples_valid(triple)]
            # print(f"Count of valid triples: {len(valid_triples)}")
            # print(valid_triples)
            # print(f"Count of valid triples: {len(valid_triples)}")
            
            #
            # Ignore triples that have a value connection
            #
            filtered_triples = []
            for triples_string in valid_triples:
                if check_value_connection and self.has_no_value_connection(triples_string) == False:
                    continue
                filtered_triples.append(triples_string)
            # print(f"Count of filtered triples: {len(filtered_triples)}")
            
            #
            # In the generated triples, replace variables that represent types/classes with their actual URIs
            #
            final_triples = []
            for triples_string in filtered_triples:
                final_triples.append(self.replace_types_for_triples(triples_string))
                
            log(f"Count of final triples: {len(final_triples)}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                
            get_kgaqa_tracker()._pe_property_path_to_triples_time += time.time() - start_time
            path_to_triples_index[key] = final_triples
            
        return path_to_triples_index[key]
    
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------
    
    def identify(self, sentence: str, reasoning_paths: List[str]):
        log(f"Grounding the reasoning paths of request: '{sentence}'", LogComponent.PATH_EXTRACTOR, LogLevel.INFO, LogType.HEADER)
        
        grounded_paths = []
        selected_triples = []
        
        for path in reasoning_paths:        
            grounded_connections = []
               
            log(f"-----------------------", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            log(f"Path: {path}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
            log(f"-----------------------", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            connections = path.split(" -> ")
            connections_og = path.split(" -> ")
            
            # find the path in the KG that connects the start and end nodes
            for i in range(len(connections) - 1):
                must_change = False
                start, end = connections[i], connections[i + 1]
                start_idx = i
                end_idx = i + 1
                class_to_class = False
                
                known_to_known = not is_entity_placeholder(end) and not is_property_description(end)
                
                # if the start is a placeholder, we don't know where to start from, so we skip it
                if is_entity_placeholder(start):
                    self.tracker._pe_unknown_start += 1
                    log(f"Start node is a placeholder: {start}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
                    continue
                # if the goal is a placeholder, we need to find the path to it
                elif is_entity_placeholder(end):
                    self.tracker._pe_unknown_goal += 1
                    candidate_paths, candidate_popularities = self.get_paths_by_neighborhood_search(sentence, start, end, to_uri=True)
                    # FIXME: This is actually wrong, we should not filter literals, it might be that our results is a literal...
                    
                    log(f"End node is a placeholder: {end}", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
                    
                    # print(candidate_paths)
                    # print(candidate_popularities)
                    # exit(0)
                    
                    if candidate_paths == []:
                        # print_colored("No predicates selected, skipping...", Colors.RED)
                        log(f"No predicates selected for {start} -> {end}, skipping...", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                        continue
                    
                    property_paths = []
                    popularities = []
                    
                    for idx in range(len(candidate_paths)):
                        candidate_path = candidate_paths[idx]
                        candidate_path += " -> PLACEHOLDER"
                        if isinstance(start, list) or self.is_entity(start):
                            property_paths.append(candidate_path)
                            popularities.append(candidate_popularities[idx])
                        elif self.is_class(start):
                            type_func = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                            if self.knowledge_graph == KnowledgeGraph.WIKIDATA:
                                type_func = "http://www.wikidata.org/prop/direct/P31"
                            type_func = uri_to_uril(type_func, self.knowledge_graph)
                            property_paths.append(f"{type_func} -> " + candidate_path)
                            popularities.append(candidate_popularities[idx])
                    
                    if len(property_paths) == 0:
                        continue          
                    
                    must_change = True       
                # we know both start and the goal
                else:
                    self.tracker._pe_known_to_known += 1
                    # get candidate paths for the connection
                    property_paths, popularities = self.get_candidate_paths(sentence, reasoning_paths, start, end)
                    if self.is_class(start) and self.is_class(end):
                        class_to_class = True
                    if property_paths == []:
                        continue
                    
                if len(property_paths) > 300:
                    property_paths = property_paths[:300]
                    popularities = popularities[:300]
                    
                if isinstance(start, list):
                    start = "VALUES ?vals {" + " ".join(start) + "} . "

                log(f"Candidate Paths with Popularities:", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
                for idx, (ppath, popularity) in enumerate(zip(property_paths, popularities), 1):
                    log(f"\t[{idx}] {ppath}  (Number of matches in the Graph: {popularity})", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                    
                # print(f"Connection {start} -> {end}")
                # print(f"START IS OF TYPE: {type(start)}")
                # print(start)

                # create triples from the property paths, the returned triples are valid in the KG
                triples_strings = []
                triples_popularities = []
                for property_path, popularity in zip(property_paths, popularities):
                    # print_colored("Property Path: " + property_path, Colors.BOLD)
                    log(f"Property Path: {property_path}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                    final_triples = self.property_path_to_triples(start, end, property_path)
                    if final_triples == []: # if the property path connects the goals through a value connection, we skip it
                        self.tracker._pe_no_triples_for_property_path += 1
                        log(f"No triples found for property path {property_path}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                        continue
                    
                    for idx, triples in enumerate(final_triples):
                        triples_strings.append(triples)
                        if class_to_class:
                            triples_popularities.append(triples_popularity(triples))
                        else:
                            triples_popularities.append(popularity)
                        # print(f"\t{idx}. ", triples)
                        log(f"\t{idx}. {triples}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                
                # if no triples were found, we skip this connection
                if len(triples_strings) == 0:
                    self.tracker._pe_no_triples_for_connection += 1
                    # print_colored("No triples found", Colors.RED)
                    log(f"No triples found for connection {start} -> {end}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                    # print_colored(f"{sentence}, {path}, {start} -> {end}", Colors.BOLD)
                    log(f"Sentence: {sentence}, Path: {path}, Connection: {start} -> {end}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                    continue
                
                # the LLM selects the triples
                enumerated_triples_string = "\n".join(f"{idx}. count of matches in the graph: {popularity} triples: {triple_string}" for idx, (triple_string, popularity) in enumerate(zip(triples_strings, triples_popularities)))
                prompt = _GENERAL_PROMPT.format(
                    preliminaries=_PROMPT_PRELIMINARIES,
                    task_explanation=_PROMPT_TASK_EXPLANATION_GROUNDING,
                    sentence=sentence,
                    path=" ".join(reasoning_paths),
                    task_prompt=_PROMPT_TASK_PROMPT_GROUNDING.format(relation=(start + " -> " + end), sentence=sentence),
                    triples=enumerated_triples_string
                )
                # print_colored(f"{prompt}", Colors.RESET)
                log(f"Prompt for grounding: {prompt}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.PROMPT)
                tries = 10
                while True and tries > 0:
                    tries -= 1
                    get_kgaqa_tracker()._pe_prompt_grounding_calls += 1
                    start_time_2 = time.time()
                    
                    generated = llm_call(self.model_id_main, prompt, 4096)
                    # print_colored(f"{generated}", Colors.BLUE)
                    log(f"LLM response for grounding: {generated}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.LLM_RESULT)
                    
                    get_kgaqa_tracker()._pe_prompt_grounding_time += time.time() - start_time_2
                    
                    if "FINAL ANSWER" in generated:
                        generated = generated.split("FINAL ANSWER")[-1]
                    
                    # FIXME, and probably change the whole logic. Instead of selecting 1, selecting multiple triples that are connected to the start and end nodes.
                    pat = r'(?<=\{).+?(?=\})'
                    index_string = re.findall(pat, generated)
                    if len(index_string) == 0:
                        continue
                    else:
                        try:
                            triples_for_connection = []
                            sample_values_for_connection = []
                            popularities_for_connection = []
                            triples_with_information_for_connection = []
                            for j in index_string:
                                if jellyfish == "":
                                    continue
                                index = int(j)
                                triples_for_connection.append(triples_strings[index])
                                if isinstance(triples_popularities[index], str) and "Sample Values:" in triples_popularities[index]:
                                    popularities_for_connection.append(triples_popularities[index].split(" | Sample Values: ")[0].strip())
                                    sample_values_for_connection.append(triples_popularities[index].split(" | Sample Values: ")[1].strip())
                                else:
                                    popularities_for_connection.append(triples_popularities[index])
                                    sample_values_for_connection.append("")

                                if must_change:
                                    samples_values = triples_popularities[index].split(" | Sample Values: ")[1].strip().split(', ')
                                    if len(samples_values) > 0:
                                        log(f"Changing connection {connections[end_idx]} to {samples_values}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
                                        connections[end_idx] = samples_values  # change the next connection to the first sample value
                                        must_change = False
                                        
                                triples_with_information_for_connection.append(TriplesWithInformation(triples_for_connection[-1], popularities_for_connection[-1], sample_values_for_connection[-1]))
                                
                            selected_triples.append(triples_for_connection)
                                                        
                            grounded_connections.append(GroundedConnection(connections_og[i], connections_og[i+1], triples_with_information_for_connection, known_to_known))
                            break
                            # index = int(index_string[0])
                            # selected_triples.append(triples_strings[index])
                        except Exception as e:
                            log(f"Error processing index string: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
                            traceback.print_exc()
                            log(f"Error converting index to int, was given: {j}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
                            continue

            if len(grounded_connections) > 0:
                grounded_paths.append(GroundedPath(path, grounded_connections))
                

        for grounded_path in grounded_paths:
            # print(grounded_path.get_formatted_information_string())
            log(grounded_path.get_formatted_information_string(), LogComponent.PATH_EXTRACTOR, LogLevel.INFO, LogType.HEADER)

        return grounded_paths
    
def triples_popularity(triples: str):       
    triples = triples_with_urils_to_triples_with_uris(triples)
    
    query = f"""
    SELECT (COUNT (DISTINCT *) as ?c) WHERE {{ 
        {triples}
    }}
    """
    log(f"triples_popularity: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)

    try:
        query_result = execute_sparql_query(query, ENDPOINT)
        results = query_result.convert()
    except Exception as e:
        log(f"Error triples_popularity: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
        log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
        return []
    
    count = []
    for result in results["results"]["bindings"]:
        count.append(result["c"]["value"])
    
    if len(count) == 0:
        return 0
        
    if count[0] == 0:
        raise ValueError("No results found for the given triples. There should be values. This was probably caused by URILs not being converted to URIs properly.")
        
    return count[0]

from tabulate import tabulate

def triples_results(triples: str):
    triples = triples_with_urils_to_triples_with_uris(triples)
    
    query = f"""
    SELECT * WHERE {{ 
        {triples}
    }}
    LIMIT 5
    """
    log(f"triples_results: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)

    try:
        query_result = execute_sparql_query(query, ENDPOINT)
        results = query_result.convert()
    except Exception as e:
        log(f"Error triples_results: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
        log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
        return [], []

    bindings = results["results"]["bindings"]
    if not bindings:
        raise ValueError("No results found for the given triples. There should be values. This was probably caused by URILs not being converted to URIs properly.")
    
    # Extract variable names
    variables = results["head"]["vars"]

    # Collect all rows with variable values
    my_results = []
    for result in bindings:
        row = [result.get(var, {}).get("value", "") for var in variables]
        for i in range(len(row)):
            if is_uri(row[i]):
                row[i] = uri_to_uril(row[i], KNOWLEDGE_GRAPH)
        my_results.append(row)

    return variables, my_results
    
class TriplesWithInformation:
    def __init__(self, triples: str, popularity: int, sample_values: List[str]):
        self.triples = triples
        self.popularity = triples_popularity(triples)
        self.variables, self.sample_values = triples_results(triples)
        
    def __str__(self):
        return f"TriplesWithInformation(triples={self.triples}, popularity={self.popularity}, sample_values={self.sample_values})"
    
class GroundedConnection:
    def __init__(self, start: str, end: str, triples: List[TriplesWithInformation], known_to_known):
        self.start = start
        self.end = end
        self.triples = triples
        self.known_to_known = known_to_known
        
class GroundedPath:
    def __init__(self, path: str, grounded_connections: List[GroundedConnection]):
        self.path = path
        self.grounded_connections = grounded_connections
        
    def get_formatted_information_string(self):
        info_string = f"### Knowledge Graph Path Analysis ###\n\n"
        info_string += f"Path:\n  {self.path}\n\n"

        for i, connection in enumerate(self.grounded_connections, start=1):
            info_string += f"--- Connection {i} ---\n"
            info_string += f"Start Node: {connection.start}\n"
            info_string += f"End Node:   {connection.end}\n\n"
            
            for j, triple_info in enumerate(connection.triples, start=1):
                info_string += f"  • Triple {j}:\n"
                info_string += f"    Pattern: {triple_info.triples}\n"
                info_string += f"    Match Count: {triple_info.popularity}\n"

                if triple_info.sample_values and triple_info.variables:
                    table_str = tabulate(triple_info.sample_values, headers=triple_info.variables, tablefmt="grid")
                    info_string += f"    Sample Values:\n{table_str}\n\n"
                else:
                    info_string += f"    Sample Values: {triple_info.sample_values}\n\n"

        return info_string


if __name__ == '__main__':    
    extractor = PathExtractor(KnowledgeGraph.WIKIDATA, model_id_main=SupportedLLMs.GPT4_1_MINI, model_id_explore=SupportedLLMs.GPT4_1_MINI)    
        
    create_logger("test", '~/', LoggingOptions.LOG_TO_CONSOLE, LogLevel.DEBUG)
    
    uri_to_uril('<http://www.wikidata.org/entity/Q154797>', KnowledgeGraph.WIKIDATA)
    
    extractor.identify("How many seats are there in the current German Bundestag ?", ["<http://www.wikidata.org/entity/Q154797_German_Bundestag> -> 'number of seats'"])    
    
    
