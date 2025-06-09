from SPARQLWrapper import SPARQLWrapper, JSON
import re

    
SPARQL_GET_CLASSES = """
SELECT DISTINCT ?class
WHERE {
    ?instance a ?class .
}
ORDER BY ?class
"""

# SPARQL_GET_CLASSES_LABELS_WIKIDATA = """
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX wdt: <http://www.wikidata.org/prop/direct/>

# SELECT DISTINCT ?class ?label
# WHERE {
#     ?item wdt:P31 ?class.
#     ?class rdfs:label ?label .
#   	FILTER(LANG(?label) = "en")
# }
# """

# SPARQL_GET_CLASSES = """
# SELECT ?class (COUNT (DISTINCT ?item) as ?c)
# WHERE {{
#     ?item a ?class.
# }}
# GROUP BY ?class
# HAVING (?c > {filter})
# ORDER BY ?class
# """

SPARQL_GET_CLASSES_LABELS_WIKIDATA = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?class ?label WHERE {{
    {{
        SELECT ?class (COUNT (DISTINCT ?item) as ?c)
        WHERE {{
            ?item wdt:P31 ?class.
        }}
        GROUP BY ?class
        HAVING (?c > {filter})
    }}
    ?class rdfs:label ?label .
    FILTER(LANG(?label) = "en")
}}
"""

# SPARQL_GET_CLASSES = """
# SELECT DISTINCT ?class
# WHERE {
#     ?instance a ?class .
# }
# ORDER BY ?class
# """

# SPARQL_GET_CLASSES_LABELS_DESCRIPTIONS_WIKIDATA = """
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX wdt: <http://www.wikidata.org/prop/direct/>

# SELECT DISTINCT ?class ?label ?desc
# WHERE {
#     ?item wdt:P31 ?class.
#     ?class rdfs:label ?label .
#   	FILTER(LANG(?label) = "en")
#     OPTIONAL {
#     	?class <http://schema.org/description> ?desc .
#         FILTER(LANG(?desc) = "en")
# 	}
# }
# """

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


def extract_labels_from_uri_default(uri: str):
    return uri.split("/")[-1]

def extract_labels_from_uri_freebase(uri: str):
    return ", ".join(uri.split("/")[-1].split("."))

def extract_labels_from_uri_dbpedia(uri: str):
    string = uri.split("/")[-1]
    labels = re.findall(r'[A-Z][a-z]*', string) # ignores numbers in label
    return " ".join(labels)

def extract_labels_from_uri_beastiary(uri: str):
    string = uri.split("#")[-1]
    labels = re.findall(r'[A-Z][a-z]*', string) # ignores numbers in label
    return " ".join(labels)    

        
def generate_class_resource_file_for_knowledge_graph(knowledge_graph, endpoint_url, extraction_function=extract_labels_from_uri_default, filter=0):
    print(f"Processing knowledge graph: {knowledge_graph}")
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setReturnFormat(JSON)
    
    print("Retrieving classes...")
    
    classes = []
    sparql.setQuery(SPARQL_GET_CLASSES.format(filter=filter))
    try:
        ret = sparql.queryAndConvert()
        # print(ret)

        for r in ret["results"]["bindings"]:
            c = r['class']['value']
            # print(c)
            classes.append(c)
    except Exception as e:
        print(e)
    
    print("Generating...")
    
    with open(knowledge_graph + f"_classes_{filter}.txt", "w") as f:
        for c in classes:
            if "Wikicat" in c:
                continue
            # labels = ", ".join(c.split("/")[-1].split(".")) # Freebase
            labels = extraction_function(c)
            f.write(f"{c} - {labels}\n")

    print("Classes written to file.")
    
def generate_class_resource_file_for_wikidata(knowledge_graph, endpoint_url, filter=0):
    print(f"Processing knowledge graph: {knowledge_graph}")
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setReturnFormat(JSON)
    
    print("Retrieving classes...")
    
    classes_labels = []
    print(SPARQL_GET_CLASSES_LABELS_WIKIDATA.format(filter=filter))
    sparql.setQuery(SPARQL_GET_CLASSES_LABELS_WIKIDATA.format(filter=filter))
    # sparql.setQuery(SPARQL_GET_CLASSES_LABELS_DESCRIPTIONS_WIKIDATA)
    try:
        ret = sparql.queryAndConvert()
        # print(ret)

        for r in ret["results"]["bindings"]:
            c = r['class']['value']
            l = r['label']['value']
            # print(c)
            classes_labels.append((c, l))
    except Exception as e:
        print(e)
    
    print("Generating...")
    
    with open(knowledge_graph + f"_classes_10.txt", "w") as f:
        for c, label in classes_labels:
            f.write(f"{c} - {label}\n")

    print("Classes written to file.")


if __name__ == '__main__':
    # generate_class_resource_file_for_knowledge_graph("dbpedia2016",
    # #                                                 #  "http://3.252.254.177:7200/repositories/wikidata-qald-10",
    # #                                                 #  "http://195.134.71.116:7200/repositories/beastiary",
    #                                                  "http://195.134.71.116:7200/repositories/dbpedia2016",
    # #                                                 #  "http://195.134.71.116:7200/repositories/freebase"
    #                                                  extraction_function=extract_labels_from_uri_dbpedia,
    #                                                  filter=20
    #                                                 )
    generate_class_resource_file_for_wikidata("wikidata", "http://3.252.254.177:7200/repositories/wikidata-qald-10", 10)
