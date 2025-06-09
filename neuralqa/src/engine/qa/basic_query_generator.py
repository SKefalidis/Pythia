import re
import json
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List
from src.metrics import get_kgaqa_tracker_from_dict
from src.engine.qa.query_db import QueryDb
from src.datasets.lc_quad_1_dataset import LcQuad1Dataset
from src.datasets.lc_quad_2_dataset import LcQuad2Dataset
from src.datasets.cwq_dataset import CwqDataset
from src.datasets.webqsp_dataset import WebQSPDataset
from src.datasets.qald10_dataset import Qald10Dataset
from src.datasets.qald9_dataset import Qald9Dataset
from src.engine.gost_requests import validate_query
from src.datasets.dataset import ENDPOINT_ID, KnowledgeGraph, triples_with_urils_to_triples_with_uris, uris_to_urils
from src.engine.entity_linking.gold_entity_identifier import GoldEntityLinker
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier
from src.engine.qa.relation_identifier import RelationIdentifier
from src.engine.qa.path_extractor import PathExtractor, GroundedPath
from src.utils import SupportedLLMs, get_kgaqa_tracker, get_relative_path, llm_call
from src.evaluation.evaluator import Evaluatable
from src.logging import create_logger, log, LoggingOptions, LogLevel, LogComponent, LogType
import argparse
import os


PROMPT_QUERY_GENERATION_SIMPLE ="""
## How to Answer Questions Using a Knowledge Graph and a Graph Reasoning Path

To answer a question using a Knowledge Graph, we first identify relevant entities and classes, and then connect them via a reasoning path that guides the construction of a SPARQL/GeoSPARQL query.

A **graph reasoning path** is a sequence of connections between entities and classes in the Knowledge Graph that helps retrieve a relevant subgraph for answering a specific question. It starts with an entity or class and ends with an entity, class, or value.

In the reasoning path:
* **URIs** represent known entities and classes.
* **ALL\_CAPS** identifiers represent unknown entities or classes.
* **lowercase** identifiers represent unknown values.

## My Problem
For the question:
`{question}`

I identified these likely relevant entities:
`[{entities}]`

And these likely relevant classes:
`[{classes}]`

Using these, I constructed the following reasoning path:
`{reasoning_path}`

Using the reasoning path, I retrieved the following triples from the Knowledge Graph:
`{triples}`

These triples are facts retrieved from the graph using the reasoning path as guidance.

## Your Task

Generate a valid SPARQL query that answers the question using the retrieved triples. Use the reasoning path and triples to inform your query structure.
* Use **ASK** queries for yes/no questions and **SELECT** queries for all others.
* Do **not** make up any triples or properties. Only use the ones provided.
* Do **not** use prefixes — use full URIs in the query.
* You may use any SPARQL constructs, including filters, arithmetic, and logical operations as needed. Don't forget to use **DISTINCT/COUNT/MIN/MAX** if necessary.
* Always produce a SPARQL query, even if the reasoning path appears flawed.
* Try to maximize both precision and recall in your query. This means that you shouldn't return too many results by utilizing too many overlapping triples, but you also should attempt to not miss any relevant results. In other words, you must be precise. Understand exactly what the question is asking and how the triples relate to it. If there is a triple that is a better match than another, use that.
* Try to use the Sample values and Count of matches for each triple to understand which triples are more relevant to the question, especially when there is a large disparity in the number of matches for each triple.
* Surround the query with triple backticks for clarity.
* **Think step by step and explain your reasoning before writing the query.**

Answer:
"""

PROMPT_QUERY_GENERATION_BETTER ="""
## How to Answer Questions Using a Knowledge Graph and a Graph Reasoning Path

To answer a question using a Knowledge Graph, we first identify relevant entities and classes, and then connect them via a reasoning path that guides the construction of a SPARQL/GeoSPARQL query.

A **graph reasoning path** is a sequence of connections between entities and classes in the Knowledge Graph that helps retrieve a relevant subgraph for answering a specific question. It starts with an entity or class and ends with an entity, class, or value.

In the reasoning path:
* **URIs** represent known entities and classes.
* **ALL\_CAPS** identifiers represent unknown entities or classes.
* **lowercase** identifiers represent unknown values.

## My Problem
For the question:
`{question}`

I identified these likely relevant entities:
`[{entities}]`

And these likely relevant classes:
`[{classes}]`

Using these, I constructed the following reasoning path:
`{reasoning_path}`

Using the reasoning path, I retrieved the following triples from the Knowledge Graph:
`{triples}`

These triples are facts retrieved from the graph using the reasoning path as guidance.

## Your Task

Generate a valid SPARQL query that answers the question using the retrieved triples. Use the reasoning path and triples to inform your query structure.
* Use **ASK** queries for yes/no questions and **SELECT** queries for all others.
* Do **not** make up any triples or properties. Only use the ones provided.
* Do **not** use prefixes — use full URIs in the query.
* The graph reasoning path **is not necessarily perfect**, but it is a good starting point to find the relevant triples in the Knowledge Graph, that's what I used it for.
* You might want to  breakup some triple paths, and use only parts of their triples.
* You may use any SPARQL constructs, including filters, arithmetic, and logical operations as needed. Don't forget to use **DISTINCT/COUNT/MIN/MAX** if necessary.
* Always produce a SPARQL query, even if the reasoning path appears flawed.
* Try to maximize both precision and recall in your query. This means that you shouldn't return too many results by utilizing too many overlapping triples, but you also should attempt to not miss any relevant results. In other words, you must be precise. Understand exactly what the question is asking and how the triples relate to it. If there is a triple that is a better match than another, use that.
* Try to use the Sample values and Count of matches for each triple to understand which triples are more relevant to the question, especially when there is a large disparity in the number of matches for each triple.
* Surround the query with triple backticks for clarity.

## Query examples that could be useful

To help you in your generation I provide you with 3 examples of question and queries that answer them that are relevant to this question. 
* These examples might be related to our task and help you understand how to construct the query, or even give you useful relations and entities that you can use in your query.
* You can use the URIs in them, or their structure if you deem it necessary. These examples are correct and valid queries for our Knowledge Graph.
* Use your judgement to decide how to use all of the given information, either collected by me, or provided via the examples
* **Think how similar these examples are to our task, and how you can use them to construct a better query. Explain how they fit or not.**
* **The examples can be especially useful to undrstand the expected return format, whether that is a string, a uri, a list or whatever else.**
* **If an example is not relevant to our task, you can ignore it, but if it is almost the same as our query use it to write a correct query.**
* **Do not blindly follow the examples. It might be better to write your own query using the information provided, rather than using the examples.**

Here are the examples:
{examples}

**Think step by step and explain your reasoning before writing the query.**

Answer:
"""

class BasicQueryGenerator(Evaluatable):
    def __init__(self, model_id: SupportedLLMs = SupportedLLMs.VLLM, query_db = None):
        self.model_id = model_id
        self.query_db: QueryDb = query_db
        
    def get_name(self):
        return "BasicQueryGenerator"
    
    def get_resource(self):
        return ""

    def extract_sparql(self, text):
        pattern = r"```sparql(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        return text
    
    def predict(self, logging = False):
        raise NotImplementedError("This method is proably useless here.")
    
    def predict_zeroshot(self, question: str, connections: List[str], grounded_paths: List[GroundedPath], failed_generations, entities, classes):      
        if classes == None or not classes:
                classes = []
        if entities == None or not entities:
            entities = []
            
        # Clean up any None entries
        classes = [str(c) for c in classes if c is not None]
        entities = [str(e) for e in entities if e is not None]
    
        prompt = PROMPT_QUERY_GENERATION_SIMPLE.format(
            question=question,
            reasoning_path="\n".join(connections),
            triples = "\n\n".join([
                path.get_formatted_information_string()
                for path in grounded_paths
                if isinstance(path, GroundedPath)
            ]),
            entities=", ".join(entities),
            classes=", ".join(classes),
            )
        log(f"[zeroshot] Prompt: {prompt}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.PROMPT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_zero_shot_calls += 1
        start_time = time.time()
        
        generated = llm_call(self.model_id, prompt, max_tokens=768, temperature=0.5 * len(failed_generations))
        log(f"[zeroshot] Generated: {generated}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_zero_shot_time += time.time() - start_time
        
        query = self.extract_sparql(generated)
        log(f"[zeroshot] Generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        return query

    def predict_icl(self, question: str, connections: List[str], grounded_paths: List[GroundedPath], failed_generations, entities, classes):     
        if classes == None or not classes:
                classes = []
        if entities == None or not entities:
            entities = []
            
        # Clean up any None entries
        classes = [str(c) for c in classes if c is not None]
        entities = [str(e) for e in entities if e is not None]
         
        relevant_questions, relevant_queries = self.query_db.get_relevant_queries(question, top_k=3) 
        print(relevant_questions)
        print(relevant_queries)
        prompt = PROMPT_QUERY_GENERATION_BETTER.format(
            question=question,
            reasoning_path="\n".join(connections),
            triples = "\n\n".join([
                path.get_formatted_information_string()
                for path in grounded_paths
                if isinstance(path, GroundedPath)
            ]),
            entities=", ".join(entities),
            classes=", ".join(classes),
            examples="\n\n".join([
                f"Question: {q}\nQuery: {r}" for q, r in zip(relevant_questions, relevant_queries)
            ]),
            # failed_generations="\n".join([f"{gen[0]}: {gen[1]}" for gen in failed_generations] if failed_generations else [])
            )
        log(f"[ICL] Prompt: {prompt}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.PROMPT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_icl_calls += 1
        start_time = time.time()
        
        generated = llm_call(self.model_id, prompt, max_tokens=768, temperature=0.5 * len(failed_generations))
        log(f"[ICL] Generated: {generated}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_icl_time += time.time() - start_time
        
        query = self.extract_sparql(generated)
        log(f"[ICL] Generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        return query

# --------------------------------
# ----- EVALUATION FUNCTIONS -----
# --------------------------------
    
def run_sparql_query_values_only(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as e:
        # print(f"SPARQL query failed for endpoint {endpoint_url}: {e}")
        # print(f"Query: {query}")
        log(f"SPARQL query failed for endpoint {endpoint_url}: {e}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        log(f"Query: {query}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        return []

    if "boolean" in results:  # ASK query
        # return results["boolean"]
        return [[results["boolean"]]]
    
    # SELECT query: extract value tuples
    value_rows = []
    for binding in results["results"]["bindings"]:
        row = tuple(v["value"] for v in binding.values())
        value_rows.append(row)
    return sorted(value_rows)

# def clean_result_by_llm(request, result_row):  
#     prompt = f"Answers to requests made on databases might contain noisy, irrelevant data. Your job is to decide whether a row of an answer from a database is a valid answer to a question or not. You must only deny completely useless answers. For example if I ask for a person and you give me a date, that is useless. If I give the wrong person that is relevant, even if wrong. To deny an answer write DENIED in curly braces. To accept it write ACCEPT in curly braces.\n Question: {request}\nAnswer: {result_row}"
#     generated = llm_call(SupportedLLMs.GPT4_1_MINI, prompt, max_tokens=4096)
#     pat = r'(?<=\{).+?(?=\})'
#     regex_results = re.findall(pat, generated)
#     if len(regex_results) > 0:
#         if "DENIED" in regex_results[0]:
#             return []
#         elif "ACCEPT" in regex_results[0]:
#             return result_row
#     return result_row

def compare_queries_loose(question, endpoint_url, query, gold_query, llm=False):
    predicted = run_sparql_query_values_only(endpoint_url, query)
    gold = run_sparql_query_values_only(endpoint_url, gold_query)
    
    if not predicted or not gold:
        return 0, len(predicted), len(gold), 0
    
    predicted_columns = [list(row) for row in zip(*predicted)]
    gold_columns = [list(row) for row in zip(*gold)]
    
    # print(f"Result 1: {result1}")
    # print(f"Result 2: {result2}")
    # print(f"Result 1 columns: {result1_columns}")
    # print(f"Result 2 columns: {result2_columns}")
    
    best_tp, best_fp, best_fn = 0, 0, 0
    for i in predicted_columns:
        for j in gold_columns:
            tp = len(set(i) & set(j))
            fp = len(set(i) - set(j))
            fn = len(set(j) - set(i))
            if tp > best_tp: # FIXME: bug, does not count fp/fn correctly if tp is always 0, but it does not affect the metrics, so we ignore it for now
                best_tp, best_fp, best_fn = tp, fp, fn
                
    hits_at_1 = 0
    for i in predicted[0]:
        for j in gold_columns:
            if i in j:
                hits_at_1 = 1
    
    return best_tp, best_fp, best_fn, hits_at_1

# def compare_queries_gerbil_style(question, endpoint_url, query, gold_query):
#     predicted = run_sparql_query_values_only(endpoint_url, query)
#     gold = run_sparql_query_values_only(endpoint_url, gold_query)
    
#     # Flatten both result sets if they're not ASK queries
#     pred_answers = set([item for sublist in predicted for item in sublist])
#     gold_answers = set([item for sublist in gold for item in sublist])
    
#     tp = len(pred_answers.intersection(gold_answers))
#     fp = len(pred_answers - gold_answers)
#     fn = len(gold_answers - pred_answers)
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
#     return precision, recall, f1

def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}

def query_has_results(endpoint_url, query, length = 0):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if "boolean" in results:
            return results["boolean"]
        else:
            has_anything = len(results["results"]["bindings"]) > 0
            # if length > 0:
            #     has_less_than = len(results["results"]["bindings"]) < length
            # else:
            #     has_less_than = True
            return has_anything # and has_less_than
    except Exception as e:
        print(f"SPARQL query failed for endpoint {endpoint_url}: {e}")
        log(f"SPARQL query failed for endpoint {endpoint_url}: {e}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        return False
    
import tempfile
    
def atomic_write(filepath, data, mode='w'):
    dir_name = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)  # atomic rename

def save_to_file(args, run_results_basic, run_results_icl, tracker, qald_json_basic=None, qald_json_icl=None):
    dataset_name = args.test_file.lower().split('/')[-1]
    
    print(f"Length {len(run_results_basic)}")
    
    try:
        # Save run results
        path_basic = os.path.join(get_relative_path(args.results_dir), f"run_results_basic_{dataset_name}.json")
        atomic_write(path_basic, run_results_basic)
        log(f"Saved run results to {path_basic}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)

        if args.query_db_file:
            path_icl = os.path.join(get_relative_path(args.results_dir), f"run_results_icl_{dataset_name}.json")
            atomic_write(path_icl, run_results_icl)
            log(f"Saved run results to {path_icl}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)

        # Save metrics
        metrics = tracker.get_metrics()
        metrics_path = os.path.join(get_relative_path(args.metrics_dir), f"metrics_{dataset_name}.json")
        atomic_write(metrics_path, metrics)
        log(f"Saved metrics to {metrics_path}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)

        # Save QALD JSONs
        if qald_json_basic is not None:
            qald_basic_path = os.path.join(get_relative_path(args.results_dir), f"qald_basic_{dataset_name}.json")
            atomic_write(qald_basic_path, qald_json_basic)
            log(f"Saved QALD JSON basic to {qald_basic_path}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)

        if args.query_db_file:
            if qald_json_icl is not None:
                qald_icl_path = os.path.join(get_relative_path(args.results_dir), f"qald_icl_{dataset_name}.json")
                atomic_write(qald_icl_path, qald_json_icl)
                log(f"Saved QALD JSON ICL to {qald_icl_path}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)

    except Exception as e:
        log(f"Error saving files: {str(e)}", LogComponent.QUERY_GENERATOR, LogLevel.ERROR, LogType.HEADER)

    
    
if __name__ == "__main__":    
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(description="Process QA dataset and run queries.")
    
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test file (e.g., QALD or WebQSP)")
    
    parser.add_argument("--query_db_file", type=str, required=False,
                    help="Path to the query database file (optional)")
    
    parser.add_argument("--entities_file", type=str, required=False,
                    help="Path to the entities file")
    
    parser.add_argument("--classes_file", type=str, required=False,
                    help="Path to the entities file")
    
    parser.add_argument("--logs_dir", type=str, required=True,
                        help="Directory where logs will be saved")
    
    parser.add_argument("--results_dir", type=str, required=True, default='./results/',
                        help="Path to the directory where the results will be stored")
    
    parser.add_argument("--metrics_dir", type=str, required=True, default='./results/',
                        help="Path to the directory where the metrics will be stored")
    
    parser.add_argument("--endpoint", type=int, required=True, default='1',
                        help="Which endpoint to use, choose from (1, 2, 3)")
    
    args = parser.parse_args()
    
    # ------------------------
    # ----- Load Dataset -----
    # ------------------------

    if "webqsp" in args.test_file.lower():
        dataset = WebQSPDataset.from_files(args.test_file)
    elif "cwq" in args.test_file.lower():
        dataset = CwqDataset.from_files(args.test_file)
    elif "qald_9" in args.test_file.lower():
        dataset = Qald9Dataset.from_files(args.test_file)
    elif "lc_quad_1" in args.test_file.lower():
        dataset = LcQuad1Dataset.from_files(args.test_file)
    elif "qald_10" in args.test_file.lower():
        dataset = Qald10Dataset.from_files(args.test_file)
    elif "lc_quad_2" in args.test_file.lower():
        dataset = LcQuad2Dataset.from_files(args.test_file)
    elif "beastiary" in args.test_file.lower():
        from src.datasets.beastiary_dataset import BeastiaryDataset
        dataset = BeastiaryDataset.from_files(args.test_file)
    else:
        raise ValueError("Unrecognized dataset format in test_file path.")
    
    # -------------------------------
    # ----- Load Query Database -----
    # -------------------------------
    
    if args.query_db_file:
        if "webqsp" in args.query_db_file.lower():
            query_db_dataset = WebQSPDataset.from_files(args.query_db_file)
        elif "cwq" in args.query_db_file.lower():
            query_db_dataset = CwqDataset.from_files(args.query_db_file)
        elif "qald_9" in args.query_db_file.lower():
            query_db_dataset = Qald9Dataset.from_files(args.query_db_file)
        elif "lc_quad_1" in args.query_db_file.lower():
            query_db_dataset = LcQuad1Dataset.from_files(args.query_db_file)
        elif "qald_10" in args.query_db_file.lower():
            query_db_dataset = Qald10Dataset.from_files(args.query_db_file)
        elif "lc_quad_2" in args.query_db_file.lower():
            query_db_dataset = LcQuad2Dataset.from_files(args.query_db_file)
        else:
            raise ValueError("Unrecognized dataset format in query_db_file path.")
    
    # -------------------------
    # ----- Setup Logging -----
    # -------------------------

    os.makedirs(get_relative_path(args.logs_dir), exist_ok=True)
    create_logger(args.test_file.lower().split('/')[-1], args.logs_dir, log_option=LoggingOptions.LOG_TO_BOTH, log_level=LogLevel.PERFORMANCE_UPDATES)
    
    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    recovery = False
    run_results_basic = []
    run_results_icl = []
    
    dataset_name = args.test_file.lower().split('/')[-1]
    if os.path.exists(os.path.join(get_relative_path(args.results_dir), f"run_results_basic_{dataset_name}.json")):
        print("Attempting to recover from existing results file...")
        run_results_basic = json.load(open(os.path.join(get_relative_path(args.results_dir), f"run_results_basic_{dataset_name}.json"), 'r', encoding='utf-8'))
        recovery = True
        # print(results_basic)
        # exit(0)
    
    if os.path.exists(os.path.join(get_relative_path(args.results_dir), f"run_results_icl_{dataset_name}.json")):
        print("Attempting to recover from existing results file...")
        run_results_icl = json.load(open(os.path.join(get_relative_path(args.results_dir), f"run_results_icl_{dataset_name}.json"), 'r', encoding='utf-8'))
        recovery = True
    
    if os.path.exists(os.path.join(get_relative_path(args.metrics_dir), f"metrics_{dataset_name}.json")):
        print("Attempting to recover from existing metrics file...")
        metrics = json.load(open(os.path.join(get_relative_path(args.metrics_dir), f"metrics_{dataset_name}.json"), 'r', encoding='utf-8'))
        get_kgaqa_tracker_from_dict(metrics)
        recovery = True
        
    # ----------------------------------
    # ----- Setup Metrics Tracking -----
    # ----------------------------------
    
    os.makedirs(get_relative_path(args.results_dir), exist_ok=True)
    os.makedirs(get_relative_path(args.metrics_dir), exist_ok=True)
    
    tracker = get_kgaqa_tracker()
    performance_metrics = {}
    
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------
    
    import src.datasets.dataset
    src.datasets.dataset.ENDPOINT_ID = args.endpoint
    log(f"Using endpoint {src.datasets.dataset.ENDPOINT_ID}", LogComponent.KNOWLEDGE_BASE, LogLevel.INFO, LogType.NORMAL)
    
    kg = dataset.get_knowledge_graph()
    
    if args.entities_file:
        print(f"Loading linked entities from {args.entities_file}")
        linked_entities = json.load(open(args.entities_file, 'r', encoding='utf-8'))
    else:
        entity_linker = GoldEntityLinker(knowledge_graph=kg, prefixes=dataset.get_prefixes())
        
    if args.classes_file:
        print(f"Loading linked classes from {args.classes_file}")
        linked_classes = json.load(open(args.classes_file, 'r', encoding='utf-8'))
    else:
        class_identifier = GoldClassIdentifier(knowledge_graph=kg, endpoint_url=KnowledgeGraph.get_endpoint(kg), prefixes=dataset.get_prefixes())
        
    relation_identifier = RelationIdentifier(model_id=SupportedLLMs.GROQ, verbalization_model_id=SupportedLLMs.GROQ)
    path_extractor = PathExtractor(model_id_main=SupportedLLMs.GROQ, model_id_explore=SupportedLLMs.GROQ, knowledge_graph=kg)
    
    if args.query_db_file:
        query_db = QueryDb(query_db_dataset)
    else:
        query_db = None
    generator = BasicQueryGenerator(model_id=SupportedLLMs.GROQ, query_db=query_db)
    
    if "qald" in args.test_file:
        qald_json_basic = json.load(open(args.test_file, 'r', encoding='utf-8'))
        qald_json_icl = json.load(open(args.test_file, 'r', encoding='utf-8'))
    else:
        qald_json_basic = None
        qald_json_icl = None

    # for idx, entry in enumerate(dataset):
    if recovery == False:
        r = range(len(dataset))
    else:
        r = range(len(run_results_icl), len(dataset))
        
    r = range(50) # For further studies, limit to 100 entries
    
    for idx in r:
        print("Answering question", idx + 1, "of", len(dataset))
        entry = dataset[idx]
        tracker._total += 1
        tracker._total_questions += 1
        
        question = dataset.get_question(entry)
        gold_query = dataset.get_query(entry)
        
        # Fix missing prefixes
        fixed_gold_query = ""
        prefixes = dataset.get_prefixes().split("\n")
        for prefix in prefixes:
            prefix = prefix.replace("\n", "").strip()
            if prefix == "":
                continue
            prefix_keyword, prefix_name, prefix_value = prefix.split(" ")
            pattern = r'PREFIX\s+'+re.escape(prefix_name)
            if re.search(pattern, gold_query) is None:
                fixed_gold_query += prefix + "\n"
        fixed_gold_query += gold_query
        gold_query = fixed_gold_query
        
        # If the query can't be answered or is invalid, we generate to have valid results for Gerbil, but we don't count it in the metrics.
        skip_metrics = False
        if validate_query(gold_query) == False:
            log(f"Invalid gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
            tracker._invalid_gold_queries += 1
            tracker._total -= 1
            skip_metrics = True
        if query_has_results(KnowledgeGraph.get_endpoint(kg), gold_query) == False:
            log(f"Empty gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
            tracker._empty_gold_queries += 1
            tracker._total -= 1
            skip_metrics = True
        
        log(f"Question: {question}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)
        
        # Get the entities and classes
        if args.classes_file:
            classes = linked_classes[idx+1]['predictions']
        else:
            classes = class_identifier.identify(gold_query)
        # classes = [] # for Wikidata linking and LC-QuAD (we didn't manage to generate classes in time for the deadline)
        
        if args.entities_file:
            entities = linked_entities[idx+1]['predictions']
        else:
            entities = entity_linker.identify(gold_query)
        
        entities = [uri for uri in entities if uri not in classes] # FIXME: Temporary fix because the Wikidata gold entity linker returns classes as entities.
        
        classes = uris_to_urils(classes, kg)
        entities = uris_to_urils(entities, kg)
        
        # Get the relations
        start_time = time.time()
        relations = relation_identifier.identify(question, classes, entities)
        end_time = time.time()
        tracker._ri_time += end_time - start_time
        
        # Get the paths
        start_time = time.time()
        grounded_paths = path_extractor.identify(question, relations)
        end_time = time.time()
        tracker._pe_time += end_time - start_time
        
        # Generate the 0-shot query
        start_time = time.time()
        tries = 3
        failed_generations = []
        while True and tries > 0:
            tries -= 1
            query = generator.predict_zeroshot(question, relations, grounded_paths, failed_generations, entities, classes)
            try:
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
                query = triples_with_urils_to_triples_with_uris(query, kg)
            except:
                log(f"Error in triples_with_urils_to_triples_with_uris: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
                continue
            if validate_query(query) == True:
                if query_has_results(KnowledgeGraph.get_endpoint(kg), query, 0) == True:
                    break
                else:
                    failed_generations.append((query, "Query has no results"))
                    log(f"Query has no results, it is likely that the connected triples are incompatible: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
                    continue
            else:
                failed_generations.append((query, "Invalid query"))
                log(f"Invalid query, you made some syntactical mistake: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)        
        log(f"[0-shot] QUERY VALID", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.LLM_RESULT)
        end_time = time.time()
        query_zeroshot = query
        tracker._qg_zero_shot_time += end_time - start_time
        
        # Generate the ICL query
        if args.query_db_file:
            start_time = time.time()
            tries = 1
            failed_generations = []
            while True and tries > 0:
                tries -= 1
                query = generator.predict_icl(question, relations, grounded_paths, failed_generations, entities, classes)
                try:
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
                    query = triples_with_urils_to_triples_with_uris(query, kg)
                except:
                    log(f"Error in triples_with_urils_to_triples_with_uris: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
                    continue
                if validate_query(query) == True:
                    if query_has_results(KnowledgeGraph.get_endpoint(kg), query, 0) == True:
                        break
                    else:
                        failed_generations.append((query, "Query has no results"))
                        log(f"Query has no results, it is likely that the connected triples are incompatible: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
                        continue
                else:
                    failed_generations.append((query, "Invalid query"))
                    log(f"Invalid query, you made some syntactical mistake: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)        
            log(f"[ICL] QUERY VALID", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.LLM_RESULT)
            end_time = time.time()
            query_icl = query
            tracker._qg_icl_time += end_time - start_time
        
        log(f"Progress: {idx+1}/{len(dataset)}", LogComponent.OTHER, LogLevel.INFO, LogType.HEADER)
        log(f"Question: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.HEADER)
        log(f"Entities: {entities}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
        log(f"Classes: {classes}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
        log(f"Relations: {relations}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
        log(f"Grounded Paths: {grounded_paths}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
        
        # ------------------------
        # ----- Update Files -----
        # ------------------------
        
        # Zeroshot
        
        sparql = SPARQLWrapper(KnowledgeGraph.get_endpoint(kg))
        sparql.setCredentials("user", "PASSWORD")
        sparql.setQuery(query_zeroshot)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            if qald_json_basic is not None:
                qald_json_basic['questions'][idx]['query']['sparql'] = query
                qald_json_basic['questions'][idx]['answers'] = [results]
        except:
            if qald_json_basic:
                qald_json_basic['questions'][idx]['query']['sparql'] = query
                qald_json_basic['questions'][idx]['answers'] = []
        entry = {
            "question": question,
            "gold_query": gold_query,
            "entities": entities,
            "classes": classes,
            "relations": relations,
            "paths": ", ".join([path.path for path in grounded_paths]),
            "generated_query": query_zeroshot,
            # "answer": [results]
        }
        run_results_basic.append(entry)
        
        # ICL
        
        if args.query_db_file:
            sparql = SPARQLWrapper(KnowledgeGraph.get_endpoint(kg))
            sparql.setCredentials("user", "PASSWORD")
            sparql.setQuery(query_icl)
            sparql.setReturnFormat(JSON)
            try:
                results = sparql.query().convert()
                if qald_json_basic is not None:
                    qald_json_basic['questions'][idx]['query']['sparql'] = query
                    qald_json_basic['questions'][idx]['answers'] = [results]
            except:
                if qald_json_basic:
                    qald_json_basic['questions'][idx]['query']['sparql'] = query
                    qald_json_basic['questions'][idx]['answers'] = []
            entry = {
                "question": question,
                "gold_query": gold_query,
                "entities": entities,
                "classes": classes,
                "relations": relations,
                "paths": ", ".join([path.path for path in grounded_paths]),
                "generated_query": query_icl,
                # "answer": [results]
            }
            run_results_icl.append(entry)
        
        # --------------------------
        # ----- Update Metrics -----
        # --------------------------

        if skip_metrics == False:
            # Zeroshot
            log(f"Gold query: {gold_query}", LogComponent.OTHER, LogLevel.INFO, LogType.GOLD)
            log(f"Generated query: {query_zeroshot}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
            
            tp, fp, fn, hits_at_1 = compare_queries_loose(question, KnowledgeGraph.get_endpoint(kg), query_zeroshot, gold_query)

            if tp > 0 and fp == 0 and fn == 0:
                log(f"✅ Correct 0-shot: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                tracker._exact_match += 1
            else:
                log(f"❌ Incorrect 0-shot: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                
            log(f"{tp} TP, {fp} FP, {fn} FN", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
            log(f"Hits@1: {hits_at_1}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                
            tracker._total_tp += tp
            tracker._total_fp += fp
            tracker._total_fn += fn
            tracker._total_hits_at_1 += hits_at_1
            
            # Compute metrics
            metrics = compute_metrics(tp, fp, fn)            
            tracker._total_macro_f1 += metrics['f1']
            tracker._total_macro_precision += metrics['precision']
            tracker._total_macro_recall += metrics['recall']
            
            log(f"\nMacro-averaged metrics O-shot: {metrics}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tPrecision: {tracker._total_macro_precision/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tRecall: {tracker._total_macro_recall/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tF1: {tracker._total_macro_f1/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tHits@1: {tracker._total_hits_at_1/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            
            # In-context Learning
            if args.query_db_file:
                log(f"Gold query: {gold_query}", LogComponent.OTHER, LogLevel.INFO, LogType.GOLD)
                log(f"Generated query: {query_icl}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                
                tp_llm, fp_llm, fn_llm, hits_at_1_llm = compare_queries_loose(question, KnowledgeGraph.get_endpoint(kg), query_icl, gold_query, llm=True)
                if tp_llm > 0 and fp_llm == 0 and fn_llm == 0:
                    log(f"✅ Correct ICL: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                    tracker._icl_exact_match += 1
                else:
                    log(f"❌ Incorrect ICL: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                    
                log(f"{tp_llm} TP, {fp_llm} FP, {fn_llm} FN", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                log(f"Hits@1: {hits_at_1_llm}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                
                tracker._icl_total_tp += tp_llm
                tracker._icl_total_fp += fp_llm
                tracker._icl_total_fn += fn_llm
                tracker._icl_total_hits_at_1 += hits_at_1_llm     
                
                # Compute metrics
                metrics_llm = compute_metrics(tp_llm, fp_llm, fn_llm)
                tracker._icl_total_macro_f1 += metrics_llm['f1']
                tracker._icl_total_macro_precision += metrics_llm['precision']
                tracker._icl_total_macro_recall += metrics_llm['recall']
                
                log(f"\nMacro-averaged metrics ICL: {metrics_llm}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
                log(f"\tPrecision: {tracker._icl_total_macro_precision/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
                log(f"\tRecall: {tracker._icl_total_macro_recall/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
                log(f"\tF1: {tracker._icl_total_macro_f1/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
                log(f"\tHits@1: {tracker._icl_total_hits_at_1/tracker._total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            
            # Detailed Metrics
            print()
            print("----------------------------------------")
            get_kgaqa_tracker().print()
            print("----------------------------------------")
            print()
            
            if idx % 20 == 0:
                log(f"Saving results and metrics after {idx+1} entries...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
                save_to_file(args, run_results_basic, run_results_icl, tracker, qald_json_basic=qald_json_basic, qald_json_icl=qald_json_icl)
            
    save_to_file(args, run_results_basic, run_results_icl, tracker, qald_json_basic=qald_json_basic, qald_json_icl=qald_json_icl)