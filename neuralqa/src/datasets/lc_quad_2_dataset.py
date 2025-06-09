import json
from src.engine.gost_requests import extract_uris
from src.datasets.dataset import Dataset, KnowledgeGraph
from tqdm import tqdm


WIKIDATA_PREFIXES = """
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX cc: <http://creativecommons.org/ns#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
    PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
    PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
    PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
    PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
    PREFIX wdsubgraph: <https://query.wikidata.org/subgraph/>
    PREFIX wdref: <http://www.wikidata.org/reference/>
    PREFIX wds: <http://www.wikidata.org/entity/statement/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
    PREFIX wdv: <http://www.wikidata.org/value/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


class LcQuad2Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__("LC-QuAD 2.0")
        
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
        
        dataset = [entry for entry in dataset if entry['question'] != '' and entry['question'] is not None]
                
        return cls(dataset)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")
        return self.dataset[idx]
    
    def get_question(self, entry):
        return entry['question']
    
    def get_query(self, entry):
        return entry['sparql_wikidata']
    
    def get_prefixes(self):
        return WIKIDATA_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraph.WIKIDATA


if __name__ == '__main__':
    dataset = LcQuad2Dataset.from_files('/home/skefalidis/qa_playground/datasets/lc_quad_2/test.json')
    
    print("Dataset:")
    print(len(dataset))
        
    for i in tqdm(dataset):
        extract_uris(WIKIDATA_PREFIXES + i['sparql_wikidata'])
        # print(i)
    