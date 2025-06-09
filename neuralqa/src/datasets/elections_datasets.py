import json
from src.datasets.dataset import Dataset, KnowledgeGraph

ELECTIONS_PREFIXES = """
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
    PREFIX strdf: <http://strdf.di.uoa.gr/ontology#>
    PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    PREFIX y2geor: <http://kr.di.uoa.gr/yago2geo/resource/>
    PREFIX y2geoo: <http://kr.di.uoa.gr/yago2geo/ontology/>
    PREFIX pnyqa: <http://pnyqa.di.uoa.gr/ontology/>
"""

ELECTIONS_CLASSES = [
    "http://pnyqa.di.uoa.gr/ontology/State",
    "http://pnyqa.di.uoa.gr/ontology/County",
    "http://pnyqa.di.uoa.gr/ontology/swamp-marsh",
    "http://pnyqa.di.uoa.gr/ontology/reservoir",
    "http://pnyqa.di.uoa.gr/ontology/stream-river",
    "http://pnyqa.di.uoa.gr/ontology/canal-ditch",
    "http://pnyqa.di.uoa.gr/ontology/lake-pond",
    "http://pnyqa.di.uoa.gr/ontology/forest",
    "http://pnyqa.di.uoa.gr/ontology/mountain"
]

class _ElectionsDataset(Dataset):
    def __init__(self, dataset, name):
        super().__init__(name)
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
                
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
        return entry['query']
    
    def get_prefixes(self):
        return ELECTIONS_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraph.ELECTIONS_KG


class ElectionsConcepts(_ElectionsDataset):
    def __init__(self, dataset):
        super().__init__(dataset, "Elections-Concepts")
        

class ElectionsPredicates(_ElectionsDataset):
    def __init__(self, dataset):
        super().__init__(dataset, "Elections-Predicates")


class ElectionsEntities(_ElectionsDataset):
    def __init__(self, dataset):
        super().__init__(dataset, "Elections-NERD")
        
    def get_entities(self, entry):
        return entry['entities']


class ElectionsQuestions(_ElectionsDataset):
    def __init__(self, dataset):
        super().__init__(dataset, "Elections-Questions")


if __name__ == '__main__':
    dataset = ElectionsConcepts.from_files('/home/skefalidis/qa_playground/datasets/elections/us_benchmark.json')
    
    print("Dataset:")
    print(len(dataset))
        
    for i in dataset:
        print(i)
        
    print(dataset.get_question(dataset[0]))
    