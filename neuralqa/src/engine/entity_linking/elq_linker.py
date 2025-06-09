from typing import List
from src.datasets.dataset import KnowledgeGraph
from src.engine.entity_linking.entity_linker import EntityLinker, wikipedia_to_wikidata
import elq.main_dense as main_dense
import argparse


class ElqLinker(EntityLinker):

    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)

        # the path where you stored the ELQ models
        models_path = "PATH_TO_ELQ_MODELS"
        config = {
            "interactive": False,
            "biencoder_model": models_path+"elq_wiki_large.bin",
            "biencoder_config": models_path+"elq_large_params.txt",
            "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "output_path": "logs/",  # logging directory
            "faiss_index": "hnsw",
            "index_path": models_path+"faiss_hnsw_index.pkl",
            "num_cand_mentions": 10,
            "num_cand_entities": 10,
            "threshold_type": "joint",
            "threshold": -3,
        }

        self.args = argparse.Namespace(**config)
        self.models = main_dense.load_models(self.args, logger=None)

    def nerd(self, question: str):
        results = main_dense.run(
            self.args, None, *self.models, test_data=[{"id": 0, "text": question.lower()}])
        uris = []
        # print(results)
        for entry in results:
            for entity in entry['pred_tuples_string']:
                wikipedia_page, mention =  'https://en.wikipedia.org/wiki/' + entity[0].replace(" ", "_"), entity[1]
                wikidata_id = wikipedia_to_wikidata(wikipedia_page)
                if wikidata_id is None:
                    continue
                uris.append(wikidata_id)
                
        if self.convert:
            uris = self.convert_to_kg(uris)
                
        return uris

    def get_name(self):
        return "ELQ"
    
    def supported_targets(self) -> List[KnowledgeGraph]:
        return [KnowledgeGraph.WIKIDATA]


if __name__ == '__main__':
    entity_linker = ElqLinker(KnowledgeGraph.WIKIDATA)
    entities = entity_linker.nerd(
        question="Which counties in California were won by the Republican party?")
    print(entities)
