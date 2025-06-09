from calendar import c
from hmac import new
import json
from tqdm import tqdm
from src.engine.gost_requests import validate_query
from src.datasets.dataset import Dataset, KnowledgeGraph


FREEBASE_PREFIXES = """
    PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX ns: <http://rdf.freebase.com/ns/>
"""


class WebQSPDataset(Dataset):
    def __init__(self, dataset):
        super().__init__("WebQSP")
        
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
        
        entries = []
        for entry in dataset['Questions']:
            new_entry = {}
            new_entry['Question'] = entry['RawQuestion']
            new_entry['Answer'] = []
            new_entry['Sparql'] = ""
            for parse in entry['Parses']:
                if new_entry['Sparql'] == "":
                    new_entry['Sparql'] = parse['Sparql']
                for answer in parse['Answers']:
                    new_entry['Answer'].append(answer['AnswerArgument'])
                new_entry['Answer'] = list(set(new_entry['Answer']))
            entries.append(new_entry)
                
        return cls(entries)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")
        return self.dataset[idx]
    
    def get_question(self, entry):
        return entry['Question']
    
    def get_query(self, entry):
        return entry['Sparql']
    
    def get_prefixes(self):
        return FREEBASE_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraph.FREEBASE


if __name__ == '__main__':
    import re
    
    dataset = WebQSPDataset.from_files('/home/skefalidis/qa_playground/datasets/webqsp/data/WebQSP.test_clean.json')
    
    print("Dataset:")
    print(len(dataset))
        
    count = 0
    for i in tqdm(dataset):
        query = i['Sparql']
        
        fixed_query = ""
        prefixes = dataset.get_prefixes().split("\n")
        for prefix in prefixes:
            prefix = prefix.replace("\n", "").strip()
            if prefix == "":
                continue
            prefix_keyword, prefix_name, prefix_value = prefix.split(" ")
            pattern = r'PREFIX\s+'+re.escape(prefix_name)
            if re.search(pattern, query) is None:
                fixed_query += prefix + "\n"
        fixed_query += query
        query = fixed_query
        
        is_valid = validate_query(query)
        if not is_valid:
            print(f"Invalid query: {i['Sparql']}")
            count += 1
        # else:
        #     print(f"Valid query: {i['Sparql']}")
    print(f"Total invalid queries: {count}")
    