from src.datasets.cwq_dataset import CwqDataset
from src.datasets.webqsp_dataset import WebQSPDataset
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from tqdm import tqdm

# dataset = WebQSPDataset.from_files('PATH_TO_FILE')
FILE = 'PATH_TO_FILE'

dataset = json.load(open('PATH_TO_FILE', 'r'))
generated = json.load(open(FILE, 'r'))
# ids = open('PATH_TO_FILE', 'r').read().splitlines()
# ids_int = [int(i.split('-')[1]) for i in ids]

# print(dataset[0])
# print(generated[0])

def run_sparql_query_values_only(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as e:
        return []

    if "boolean" in results:  # ASK query
        return [[results["boolean"]]]
    
    # SELECT query: extract value tuples
    value_rows = []
    for binding in results["results"]["bindings"]:
        row = tuple(v["value"] for v in binding.values())
        value_rows.append(row)
    return sorted(value_rows)

def get_name_query(entity, endpoint_url):
    query = f"""
    SELECT ?name WHERE {{
        <{entity}> <http://rdf.freebase.com/ns/type.object.name> ?name .
    }}
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
    except Exception as e:
        return []
    
    # SELECT query: extract value tuples
    value_rows = []
    for binding in results["results"]["bindings"]:
        row = tuple(v["value"] for v in binding.values())
        value_rows.append(row)
    return sorted(value_rows)
    

def compare_queries_loose(endpoint_url, query, gold_query):
    predicted = run_sparql_query_values_only(endpoint_url, query)
    gold = run_sparql_query_values_only(endpoint_url, gold_query)
    
    if not predicted or not gold:
        return 2
    
    predicted_columns = [list(row) for row in zip(*predicted)]
    gold_columns = [list(row) for row in zip(*gold)]
                
    for row in predicted:
        for item in row:
            for j in gold_columns:
                if item in j:
                    return 1
                
    # print(len(predicted), len(gold))
    if len(predicted) > 50:
        return 2
    
    # print("Predicted:", predicted)
    # print("Gold:", gold)
    
    predicted_names = [get_name_query(item, endpoint_url) for row in predicted for item in row]
    # print(predicted_names)
    gold_names = [get_name_query(item, endpoint_url) for row in gold for item in row]
    # print(gold_names)
    # exit(0)
    
    # for name in predicted_names:
    #     for n in name:
    #         for j in gold_names:
    #             for m in j:
    #                 print(f"Comparing {n[0]} with {m[0]}")
    #                 if n[0] in m[0]:
    #                     print(f"Found name {n[0]} in {m[0]}")
    #                     return 1
            
                
    for row in predicted:
        for item in row:
            names = get_name_query(item, endpoint_url)
            # print(names)
            for name in names:
                for j in gold:
                    if name in j: # works correctly because j is a tuple
                        print(f"Found name {name} in {j}")
                        return 1
                    # else:
                        # print(f"Name {name} not found in {gold}")
                    
    for row in gold:
        for item in row:
            names = get_name_query(item, endpoint_url)
            # print("Name 2:", names)
            for name in names:
                name = name[0] if isinstance(name, tuple) else name
                for j in predicted:
                    if name in j:
                        print(f"Found name {name} in {j}")
                        return 1
                        
    # print(f"{predicted} vs {gold}")
    # print(f"{get_name_query(predicted[0][0], endpoint_url)} vs {get_name_query(gold[0][0], endpoint_url)}")
    
    return 0

skipped = 0
hits = 0
total = 0
wrong_queries = []

# dictionary = {j['QuestionId']: idx for idx, j in enumerate(dataset['Questions'])}

# print(dictionary)

# print(ids)
for i in tqdm(range(len(generated))):
# for i in tqdm(ids):
    total += 1
    # idx = dictionary[i]
    idx = i
    gold_query = generated[idx]['gold_query']
    generated_query = generated[idx]['generated_query']
    
    result = compare_queries_loose('REPO_LINK', generated_query, gold_query)
    if result == 2:
        skipped += 1
        continue
    if result == 0:
        wrong_queries.append(i)
    hits += result
print(f"Performance: {hits/(total - skipped):.2f} (hits: {hits}, total: {total - skipped})")
print(FILE)
    
# print(f"Total hits: {hits} out of {len(generated)}")
# print(f"Wrong queries: {len(wrong_queries)}")
# for i in wrong_queries:
#     print(f"\t{i}")