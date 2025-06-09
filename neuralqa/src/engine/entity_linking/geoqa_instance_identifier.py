import requests
import json

from src.engine.entity_linking.entity_linker import EntityLinker


class GeoqaInstanceIdentifier(EntityLinker):
    
    def __init__(self, knowledge_graph):
        super().__init__(knowledge_graph)

    def geoqa_send_request(self, question: str, url: str) -> requests.Response:
        data = {
            "question": question,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response

    def nerd(self, question: str):
        response = self.geoqa_send_request(
            question, 'http://localhost:12345/startquestionansweringwithtextquestion').json()
        if response['status'] == 200:
            # instances = response['instances']
            instances = response['instancesSelected']
            # instances = [i.replace('http://yago-knowledge.org/resource/',
            #                        'https://en.wikipedia.org/wiki/') for i in instances]
            return instances
        else:
            return []


if __name__ == '__main__':
    entity_linker = GeoqaInstanceIdentifier()
    entities = entity_linker.nerd(
        question="Which counties in California were won by the Republican party?")
    print(entities)
