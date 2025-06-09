from typing import List
from src.datasets.dataset import KnowledgeGraph
from src.engine.entity_linking.entity_linker import EntityLinker
from multiel import BELA


class Bela(EntityLinker):

    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)
        self.bela = BELA(device="cuda:0")
        
    def nerd(self, question: str, debug: bool = False, logging: bool = False):
        results = self.bela.process_batch([question])
        # print(results)
        uris = []
        for entry in results:
            for entity in entry['entities']:
                uris.append(f"http://www.wikidata.org/entity/{entity}")
        
        if self.convert:
            uris = self.convert_to_kg(uris)        
        
        if logging == False:
            return uris
        else:
            return uris, None
    
    def get_name(self):
        return "BELA"
    
    def supported_targets(self) -> List[KnowledgeGraph]:
        return [KnowledgeGraph.WIKIDATA]


if __name__ == '__main__':
    entity_linker = Bela()
    entities = entity_linker.nerd(
        question="Which counties in California were won by the Republican party?")
    print(entities)
