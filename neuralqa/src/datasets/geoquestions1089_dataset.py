import json
from src.datasets.dataset import Dataset, KnowledgeGraph

GEOQUESTIONS1089_PREFIXES = """
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
"""

class Geoquestions1089Dataset(Dataset):
    def __init__(self, dataset, keys):
        super().__init__("Geoquestions1089")
        
        self.dataset = dataset
        self.keys = keys

        # Replace 'same-as' references with real values
        for idx, entry in enumerate(self):
            for key, value in entry.items():
                if 'same-as' in value:
                    same_as_key = value.split(':')[1].strip()
                    if same_as_key in self.dataset:
                        entry[key] = self.dataset[same_as_key][key]
                    else:
                        raise KeyError(f"Original entry with key {same_as_key} not found in dataset")

    @classmethod 
    def from_files(cls, file_path: str, answers_path: str = None):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        keys = list(dataset.keys())
        data_file.close()

        # Load the gold answers if available
        if answers_path:
            data_file = open(answers_path)
            answers = json.load(data_file)
            data_file.close()
            
            for key in answers:
                if key not in dataset:
                    raise KeyError(f"Key {key} not found in dataset")
                dataset[key]['Answer'] = answers[key] if answers[key] else ""
                
        return cls(dataset, keys)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            raise IndexError("Index out of range")
        return self.dataset[self.keys[idx]]
    
    def __setitem__(self, idx, value):
        if idx >= len(self.keys):
            raise IndexError("Index out of range")
        self.dataset[self.keys[idx]] = value
        
    @staticmethod
    def get_question(entry):
        return entry['Question']
    
    @staticmethod
    def get_query(entry):
        return entry['Query']
    
    @staticmethod
    def get_prefixes():
        return GEOQUESTIONS1089_PREFIXES
    
    @staticmethod
    def get_knowledge_graph():
        return KnowledgeGraph.YAGO2geo
    
    def y2geo_subset(self):
        return Geoquestions1089Dataset(self.dataset, [self.keys[idx] for idx in range(0, 894) if idx < len(self.keys)])
    
    def c_subset(self):
        return Geoquestions1089Dataset(self.dataset, [self.keys[idx] for idx in range(0, 1017) if idx < len(self.keys)])
    
    def w_subset(self):
        return Geoquestions1089Dataset(self.dataset, [self.keys[idx] for idx in range(1017, 1089) if idx < len(self.keys)])
    
    def category_subset(self, categories: str):
        if 'Category' not in self[0].keys():
            raise KeyError("Category key not found in dataset")
        
        CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        if categories not in CATEGORIES:
            raise ValueError(f"Invalid category. Expected one of {CATEGORIES}, got {categories}")
        
        indices = []
        for idx, entry in enumerate(self):
            if entry['Category'] in categories:
                indices.append(idx)
        return Geoquestions1089Dataset(self.dataset, [self.keys[idx] for idx in indices if idx < len(self.keys)])


if __name__ == '__main__':
    dataset = Geoquestions1089Dataset.from_files('/home/skefalidis/GeoQuestions1089/GeoQuestions1089.json', 
                                                #  '/home/skefalidis/GeoQuestions1089/GeoQuestions1089_answers.json'
                                                 )
    
    print("Dataset:")
    print(len(dataset))
    
    print("Subsets:")
    print(len(dataset.c_subset()))
    print(len(dataset.w_subset()))
    
    print("Category A:")
    print(len(dataset.category_subset('A')))
    print(len(dataset.c_subset().category_subset('A')))
    print(len(dataset.w_subset().category_subset('A')))
    
    print("Intersection of subsets:")
    print(len(dataset.c_subset().w_subset()))
        
    for i in dataset.w_subset():
        print(i)
    