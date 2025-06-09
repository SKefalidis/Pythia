import requests
import json

GOST_ENDPOINT = 'http://localhost:9090/'


def gost_request(query: str, endpoint: str, chance=-1.0):
    data = {
        "query": query,
        "chance": chance
    }

    headers = {
        'Content-Type': 'application/json'
    }
    return requests.post(GOST_ENDPOINT + endpoint, headers=headers, data=json.dumps(data))


def validate_query(query: str):
    response = gost_request(query, 'validate-api')
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
        return None


def format_query(query: str):
    response = gost_request(query, 'format')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


def materialize_query(query: str):
    response = gost_request(query, 'materialize-api')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None


def extract_uris(query: str):
    response = gost_request(query, 'uris')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        print(query)
        return None


def extract_predicates(query: str):
    response = gost_request(query, 'predicates')
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.text)
        return None
