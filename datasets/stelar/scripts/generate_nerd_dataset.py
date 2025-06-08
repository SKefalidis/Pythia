# from transformers import pipeline
# import torch
import re
import requests
import json
import random

# -----------------------------------
# ----- GET INSTANCES FOR TYPES -----
# -----------------------------------

def graphdb_send_request(query, accept_format='application/sparql-results+json'):
    """
    Sends a SPARQL query to a GraphDB endpoint.

    :param query: SPARQL query to be sent
    :param endpoint_url: URL of the GraphDB SPARQL endpoint
    :param accept_format: Desired response format (default is JSON)
    :return: Response from the endpoint
    """
    GRAPHDB_ENDPOINT = 'http://195.134.71.116:7200/repositories/stelar3'
    username = 'user'
    password = 'omn1ss1@h'
    
    headers = {
        'Accept': accept_format,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'query': query
    }

    response = requests.post(GRAPHDB_ENDPOINT, headers=headers, data=data,
                             auth=requests.auth.HTTPBasicAuth(username, password))

    if response.status_code == 200:
        if accept_format == 'application/sparql-results+json':
            return response.json()
        else:
            return response.text
    else:
        response.raise_for_status()

GET_INSTANCES_QUERY = """
    SELECT ?s WHERE {{
        ?s a <{type}>
    }} LIMIT 100
"""

CLASSES = [
    "http://www.w3.org/ns/dcat#Dataset", 
    "http://www.w3.org/ns/dcat#Distribution",
    "http://stelar-project.eu/klms#Organization",
    "http://stelar-project.eu/klms#User",
    "http://stelar-project.eu/klms#WorkflowExecution",
    "http://stelar-project.eu/klms#TaskExecution"
]

class_instances = {}
for c in CLASSES:
    results = graphdb_send_request(GET_INSTANCES_QUERY.format(type=c))
    instances = [r['s']['value'] for r in results['results']['bindings']]
    class_instances[c] = instances
    
# print(class_instances)

# -------------------------------------------
# ----- GENERATE SYNTHETIC NERD DATASET -----
# -------------------------------------------

import openai

def get_instance(c: str):
    if c in class_instances.keys() and class_instances[c]:
        instance = random.choice(class_instances[c])
        return instance, instance.split("/")[-1]
    return None

def placeholder_to_uri(c: str):
    if c == "DATASET":
        return "http://www.w3.org/ns/dcat#Dataset"
    if c == "DISTRIBUTION":
        return "http://www.w3.org/ns/dcat#Distribution"
    if c == "ORGANIZATION":
        return "http://stelar-project.eu/klms#Organization"
    if c == "USER":
        return "http://stelar-project.eu/klms#User"
    if c == "WORKFLOW":
        return "http://stelar-project.eu/klms#WorkflowExecution"
    if c == "TASK":
        return "http://stelar-project.eu/klms#TaskExecution"

GENERATION_PROMPT = """
    You are given an ontology file and some classes of this ontology. 
    You are tasked with creating 100 synthetic requests that use instances of these classes. Since you will not be able to know the instances just add placeholders.
    Specifically:
    "http://www.w3.org/ns/dcat#Dataset" placeholder {{DATASET}}
    "http://www.w3.org/ns/dcat#Distribution" placeholder {{DISTRIBUTION}}
    "http://stelar-project.eu/klms#Organization" placeholder {{ORGANIZATION}}
    "http://stelar-project.eu/klms#User" placeholder {{USER}}
    "http://stelar-project.eu/klms#WorkflowExecution" placeholder {{WORKFLOW}}
    "http://stelar-project.eu/klms#TaskExecution" placeholder {{TASK}}
    
    You can create complex requests that utilize multiple classes and/or instances in the same request. You must include at least one instance in each request.
    
    Examples:
    Is {{USER}} a member of {{ORGANIZATION}}?
    How many datasets have been published by {{ORGANIZATION}}?
    Which task execution have ran as part of {{WORKFLOW}}?
    
    Your 100 synthetic requests:
"""

# print(GENERATION_PROMPT)

file = open('nerd_dataset_placeholders.txt', 'r')
dataset = []
for line in file:
    classes = re.findall(r'\{\{(.*?)\}\}', line)
    instances = []
    for c in classes:
        instance, identifier = get_instance(placeholder_to_uri(c))
        line = line.replace("{{" + c + "}}", identifier, 1)
        instances.append(instance)
    line = line.replace("\n", "")
    print(line)
    dataset.append({
        'question': line,
        'instances' : instances
    })
    
with open("stelar_nerd.json", "w") as outfile:
    json.dump(dataset, outfile, indent=4, sort_keys=False)