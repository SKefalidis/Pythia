from typing import List
from src.datasets.cwq_dataset import CwqDataset
from src.datasets.dataset import KnowledgeGraph
from src.engine.entity_linking.entity_linker import EntityLinker
from src.engine.gost_requests import extract_uris
from SPARQLWrapper import SPARQLWrapper, JSON


def generic_uri_is_entity(uri: str, endpoint: str) -> bool:
    sparql = SPARQLWrapper(endpoint)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setReturnFormat(JSON)
    
    # print(f"URI: {uri}")

    # Is the URI used as a predicate?
    sparql.setQuery(f"""
        ASK WHERE {{
            ?s <{uri}> ?o .
        }}
    """)
    if sparql.query().convert()['boolean']:
        return False 

    # Is the URI used as an rdf:type?
    sparql.setQuery(f"""
        ASK WHERE {{
            ?s a <{uri}> .
        }}
    """)
    if sparql.query().convert()['boolean']:
        return False
    
    # print(f"URI: {uri} is not a predicate or rdf:type")
    
    return True

def elections_uri_is_entity(uri: str) -> bool:
    return y2geo_uri_is_entity(uri)

def stelar_uri_is_entity(uri: str) -> bool:
    return generic_uri_is_entity(uri, KnowledgeGraph.get_endpoint(KnowledgeGraph.STELAR_KG))

def terraq_uri_is_entity(uri: str) -> bool:
    return y2geo_uri_is_entity(uri)

def beastiary_uri_is_entity(uri: str) -> bool:
    return generic_uri_is_entity(uri, KnowledgeGraph.get_endpoint(KnowledgeGraph.BEASTIARY_KG))

def y2geo_uri_is_entity(uri: str) -> bool:
    return "/resource/" in uri and '/has' not in uri

def wikidata_uri_is_entity(uri: str) -> bool:
    return "http://www.wikidata.org/entity/" in uri # FIXME: WRONG WRONG WRONG

def dbpedia_uri_is_entity(uri: str) -> bool:
    return "http://dbpedia.org/resource" in uri

def freebase_uri_is_entity(uri: str) -> bool:
    # print(f"URI: {uri}")
    if generic_uri_is_entity(uri, KnowledgeGraph.get_endpoint(KnowledgeGraph.FREEBASE)) == False:
        return False
    return "http://rdf.freebase.com/ns/" in uri


class GoldEntityLinker(EntityLinker):
    
    def __init__(self, knowledge_graph: str, prefixes: str):
        super().__init__(knowledge_graph)
        self.prefixes = prefixes
        if self.knowledge_graph == KnowledgeGraph.YAGO2geo:
            self.entity_identification_function = y2geo_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.WIKIDATA:
            self.entity_identification_function = wikidata_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.DBPEDIA or self.knowledge_graph == KnowledgeGraph.DBPEDIA10:
            self.entity_identification_function = dbpedia_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.FREEBASE:
            self.entity_identification_function = freebase_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.ELECTIONS_KG:
            self.entity_identification_function = y2geo_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.STELAR_KG:
            self.entity_identification_function = stelar_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.TERRAQ_KG:
            self.entity_identification_function = terraq_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraph.BEASTIARY_KG:
            self.entity_identification_function = beastiary_uri_is_entity
    
    def nerd(self, query: str):
        all_uris = extract_uris(self.prefixes + "\n" + query)
        all_uris = all_uris.split("\n")
        entity_uris = []
        for u in all_uris:
            if u == "":
                continue
            if self.entity_identification_function(u) == True:
                entity_uris.append(u)
        return list(set(entity_uris))
    
    def supported_targets(self) -> List[KnowledgeGraph]:
        return [kg for kg in KnowledgeGraph] 
    
    def get_name(self):
        return self.knowledge_graph + " Gold Entity Linker"
    
    def get_resource(self):
        return "GoST"
    
    
if __name__ == "__main__":
    # Test the GoldEntityLinker
    gold_entity_linker = GoldEntityLinker(KnowledgeGraph.FREEBASE, CwqDataset.get_prefixes())
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE { 
        FILTER (?x != ?c) FILTER (!isLiteral(?x) || lang(?x) = '' || langMatches(lang(?x), 'en')) 
        ?c ns:organization.organization.leadership ?k . ?k ns:organization.leadership.person ns:m.0hhv_6h .  
        ?c ns:sports.sports_team.championships ?x . ?x ns:time.event.start_date ?sk0 . 
    } 
    ORDER BY DESC(xsd:datetime(?sk0)) LIMIT 1
    """
    print(gold_entity_linker.nerd(query))