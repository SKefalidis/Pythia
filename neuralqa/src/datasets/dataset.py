from enum import Enum
from typing import Dict
from torch.utils.data import Dataset as TorchDataset
from abc import abstractmethod
import requests
import traceback
import re
from src.logging import log, LogComponent, LoggingOptions, LogLevel, LogType

class Dataset(TorchDataset):
    def __init__(self, name):
        self._name = name
        
    def get_name(self):
        return self._name
    
    @abstractmethod
    def get_question(self, entry):
        raise NotImplementedError("get_question method not implemented")
    
    @abstractmethod
    def get_query(self, entry):
        raise NotImplementedError("get_question method not implemented")
    
    @abstractmethod
    def get_prefixes(self):
        raise NotImplementedError("get_prefixes method not implemented")
    
    @abstractmethod
    def get_knowledge_graph(self):
        raise NotImplementedError("get_prefixes method not implemented")
    
    def __str__(self):
        return self.get_name()
    
    
ENDPOINT_ID = 6
    
class KnowledgeGraph(Enum):
    YAGO2geo = "YAGO2geo"
    ELECTIONS_KG = "Elections_KG"
    STELAR_KG = "STELAR_KG"
    TERRAQ_KG = "TERRAQ_KG"
    FREEBASE = "Freebase"
    DBPEDIA = "DBpedia"
    DBPEDIA10 = "DBpedia10"
    WIKIDATA = "Wikidata"
    BEASTIARY_KG = "Beastiary_KG"
    
    def get_endpoint(kg):
        endpoint_address = "NONE"
        if ENDPOINT_ID == 1:
            endpoint_address = 'http://54.74.48.25:7200/repositories/'
        elif ENDPOINT_ID == 2:
            endpoint_address = 'http://18.201.59.158:7200/repositories/'
        elif ENDPOINT_ID == 3:
            endpoint_address = 'http://34.240.58.23:7200/repositories/'
        elif ENDPOINT_ID == 4:
            endpoint_address = 'http://54.155.247.97:7200/repositories/'
        elif ENDPOINT_ID == 5:
            endpoint_address = 'http://54.74.25.134:7200/repositories/'
        elif ENDPOINT_ID == 6:
            endpoint_address = 'http://34.244.106.40:7200/repositories/'
            
        if kg == KnowledgeGraph.BEASTIARY_KG:
            return endpoint_address + "beastiary"
        elif kg == KnowledgeGraph.DBPEDIA:
            return endpoint_address + "dbpedia2016"
        elif kg == KnowledgeGraph.DBPEDIA10:
            return endpoint_address + "dbpedia10"
        elif kg == KnowledgeGraph.ELECTIONS_KG:
            return endpoint_address + "pnyqa_kg_2"
        elif kg == KnowledgeGraph.STELAR_KG:
            return endpoint_address + "stelar3"
        elif kg == KnowledgeGraph.TERRAQ_KG:
            return endpoint_address + "da4dte_final"
        elif kg == KnowledgeGraph.YAGO2geo:
            return endpoint_address + "yago2geo"
        elif kg == KnowledgeGraph.FREEBASE:
            return endpoint_address + "freebase"
        elif kg == KnowledgeGraph.WIKIDATA:
            return endpoint_address + "wikidata-qald-10"
        
    def get_ontology_endpoint(kg):
        endpoint_address = "NONE"
        if ENDPOINT_ID == 1:
            endpoint_address = 'http://54.74.48.25:7200/repositories/'
        elif ENDPOINT_ID == 2:
            endpoint_address = 'http://18.201.59.158:7200/repositories/'
        elif ENDPOINT_ID == 3:
            endpoint_address = 'http://34.240.58.23:7200/repositories/'
        elif ENDPOINT_ID == 4:
            endpoint_address = 'http://54.155.247.97:7200/repositories/'
        elif ENDPOINT_ID == 5:
            endpoint_address = 'http://18.201.180.130:7200/repositories/'
        elif ENDPOINT_ID == 6:
            endpoint_address = 'http://34.244.106.40:7200/repositories/'
        
        if kg == KnowledgeGraph.BEASTIARY_KG:
            return endpoint_address + "beastiary_ontology"
        elif kg == KnowledgeGraph.DBPEDIA:
            return endpoint_address + "dbpedia_ontology"
        elif kg == KnowledgeGraph.DBPEDIA10:
            return endpoint_address + "dbpedia10-ontology"
        elif kg == KnowledgeGraph.ELECTIONS_KG:
            return endpoint_address + "pnyqa_ontology"
        elif kg == KnowledgeGraph.STELAR_KG:
            return endpoint_address + "stelar_ontology"
        elif kg == KnowledgeGraph.TERRAQ_KG:
            return endpoint_address + "terraq_ontology"
        elif kg == KnowledgeGraph.YAGO2geo:
            return endpoint_address + "yago2geo_ontology"
        elif kg == KnowledgeGraph.FREEBASE:
            return endpoint_address + "freebase_ontology"
        elif kg == KnowledgeGraph.WIKIDATA:
            return endpoint_address + "wikidata_ontology"
        
uri_to_uril_map = {}
uril_to_uri_map = {}
    
def is_uri(s):
    if s is None or not isinstance(s, str):
        return False
    return "http://" in s or "https://" in s
    
def uri_to_uril(uri: str, kg):# -> Any:
    # print(f"uri_to_uril: @{uri}@")
    if not is_uri(uri):
        return uri
    
    # print(uri_to_uril_map)
    # print(uril_to_uri_map)
    
    # print("uri_to_uril called with: ", uri)
    
    og_uri = uri
    
    had_brackets = False
    if uri[0] == "<" and uri[-1] == ">":
        uri = uri[1:-1]
        had_brackets = True
        
    if uri in uril_to_uri_map:
        return og_uri
    
    if uri in uri_to_uril_map:
        # print(f"in map @{uri}@")
        if had_brackets:
            return "<" + uri_to_uril_map[uri] + ">"
        else:
            return uri_to_uril_map[uri]
    
    id = uri.split("/")[-1]
    
    if kg == KnowledgeGraph.WIKIDATA:
        label = get_wikidata_label(id)
        if label is not None:
            # print("got label")
            label = label.replace(" ", "_")
            uri_to_uril_map[uri] = uri + "_" + label
            uril_to_uri_map[uri + "_" + label] = uri
        else:
            # print("no label")
            uri_to_uril_map[uri] = uri
            uril_to_uri_map[uri] = uri
    elif kg == KnowledgeGraph.FREEBASE:
        # print(id)
        if len(id) > 2 and (id[1] == '.' or id[2] == '.'):
            label = get_freebase_label(uri)
            # print(label)
            if label is not None :
                # print("got label")
                label = label.replace(" ", "_")
                uri_to_uril_map[uri] = uri + "_" + label
                uril_to_uri_map[uri + "_" + label] = uri
                # print(f"uri_to_uril_map[{uri}] = {uri_to_uril_map[uri]}")
                # print(f"uril_to_uri_map[{uri + '_' + label}] = {uril_to_uri_map[uri + '_' + label]}")
            else:
                # print("no label")
                uri_to_uril_map[uri] = uri
                uril_to_uri_map[uri] = uri
        else:
            # print("no label 2")
            uri_to_uril_map[uri] = uri
            uril_to_uri_map[uri] = uri
    else:
        # print("no label 3")
        uri_to_uril_map[uri] = uri
        uril_to_uri_map[uri] = uri

    # print(f"uri_to_uril done: {uri}")
    
    if had_brackets:
        return "<" + uri_to_uril_map[uri] + ">"
    else:
        return uri_to_uril_map[uri]

def uris_to_urils(uris: list, kg):
    urils = []
    for uri in uris:
        uril = uri_to_uril(uri, kg)
        urils.append(uril) 
    return urils

def uril_to_uri(uril: str):
    # print(f"uril_to_uri: @{uril}@")
    if not is_uri(uril):
        return uril
    
    # print(uri_to_uril_map)
    # print(uril_to_uri_map)
    
    if uril in uri_to_uril_map:
        if isinstance(uril, str) and re.search(r"[QP][0-9]+", uril):
            log(f"Warning: Should have been URIL @{uril}@, is URI.", LogComponent.OTHER, LogLevel.WARNING, LogType.NORMAL)
        # traceback.print_stack()
        return uril        
    
    had_brackets = False
    if uril[0] == "<" and uril[-1] == ">":
        # print("brackets")
        uril = uril[1:-1]
        had_brackets = True
    
    if uril in uril_to_uri_map:
        # print("find")
        if had_brackets:
            return "<" + uril_to_uri_map[uril] + ">"
        else:
            return uril_to_uri_map[uril]
    else:
        # print("not found")
        # If not found, return the original URIL as is. It has not been inserted, therefore it has not been converted yet.
        if had_brackets:
            return "<" + uril + ">"
        else:
            return uril  
        #raise ValueError(f"URI with label @{uril}@ not found in the mapping.")
    
def urils_to_uris(urils: list,):
    uris = []
    for uril in urils:
        uri = uril_to_uri(uril)
        uris.append(uri) 
    return uris

# def triples_with_urils_to_triples_with_uris(triples: str, kg):
#     triples_with_uris = []
#     for triple in triples.split("\n"):
#         split_elements = triple.split(" ")
#         for i in range(len(split_elements)):
#             if is_uri(split_elements[i]):
#                 split_elements[i] = uril_to_uri(split_elements[i])
#         triples_with_uris.append(" ".join(split_elements))
#     return "\n".join(triples_with_uris)

def triples_with_urils_to_triples_with_uris(triples: str, kg=None):
    def replace_uri(match):
        uri = match.group(1)
        return f"<{uril_to_uri(uri)}>"

    # Replace each URI inside < > using regex and the replace_uri function
    return re.sub(r'<([^>]*)>', replace_uri, triples)

# def triples_with_uris_to_triples_with_urils(triples: str, kg):
#     triples_with_urils = []
#     for triple in triples.split("\n"):
#         split_elements = triple.split(" ")
#         for i in range(len(split_elements)):
#             if is_uri(split_elements[i]):
#                 split_elements[i] = uri_to_uril(split_elements[i], kg)
#         triples_with_urils.append(" ".join(split_elements))
#     return "\n".join(triples_with_urils)

def triples_with_uris_to_triples_with_urils(triples: str, kg):
    def replace_uri(match):
        uri = match.group(1)
        return f"<{uri_to_uril(uri, kg)}>"

    # Replace each URI inside < > using regex and the replace_uri function
    return re.sub(r'<([^>]*)>', replace_uri, triples)
    
def do_nothing(string):
    return string
    
import re
import requests

def get_wikidata_label(qid, lang="en"):
    if not isinstance(qid, str) or not re.fullmatch(r"[QP]\d+", qid):
        return None

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
        "props": "labels",
        "languages": lang
    }
    headers = {
        "User-Agent": "QuestionAnsweringQidTranslator/1.0 (https://ai.di.uoa.gr/; skefalidis@di.uoa.gr)"
    }

    # print(f"Fetching label for {qid} from {url}")
    try:
        response = requests.get(url, params=params, headers=headers, timeout=(5, 5))
        # print(response.text)
        response.raise_for_status()
        data = response.json()
        return data["entities"][qid]["labels"][lang]["value"]
    except (KeyError, requests.exceptions.RequestException) as e:
        print(f"Error: {e}")
        return None
    
from SPARQLWrapper import SPARQLWrapper, JSON
    
def get_freebase_label(uri):
    # query = f"""
    # SELECT ?tailEntity WHERE {{
    #     {{
    #         <{uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?tailEntity .
    #         FILTER (lang(?tailEntity) = "en")
    #     }}
    #     UNION
    #     {{
    #         <{uri}> <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    #     }}
    # }}
    # """
    
    query = f"""
    SELECT ?tailEntity WHERE {{
            <{uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?tailEntity .
            FILTER (lang(?tailEntity) = "en")
    }}
    """
    
    sparql = SPARQLWrapper(KnowledgeGraph.get_endpoint(KnowledgeGraph.FREEBASE))
    sparql.setCredentials("user", "PASSWORD")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    
    try:
        results = sparql.query().convert()
        if len(results["results"]["bindings"]) > 0:
            return results["results"]["bindings"][0]["tailEntity"]["value"]
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return "Unnamed Entity"
    
    
if __name__ == "__main__":
    uri = "http://rdf.freebase.com/ns/m.03_r3"
    
    label = get_freebase_label(uri)
    
    print(f"Label for {uri}: {label}")