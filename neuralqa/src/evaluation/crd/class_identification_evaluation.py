import argparse
import yaml
import torch
import gc
from tqdm import tqdm
from datetime import datetime

from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from src.utils import SupportedLLMs, get_relative_path
from src.datasets.dataset import KnowledgeGraph
from src.engine.class_identifier.dev import MixClassIdentifier, MixClassIdentifier3
from src.datasets.lc_quad_2_dataset import LcQuad2Dataset
from src.datasets.lc_quad_1_dataset import LcQuad1Dataset
from src.datasets.beastiary_dataset import BeastiaryDataset
from src.datasets.qald10_dataset import Qald10Dataset
from src.datasets.qald9_dataset import Qald9Dataset
from src.datasets.cwq_dataset import CwqDataset
from src.datasets.webqsp_dataset import WebQSPDataset
from src.datasets.elections_datasets import ElectionsConcepts
from src.datasets.geoquestions1089_dataset import Geoquestions1089Dataset
from src.evaluation.evaluator import Evaluator

from src.engine.class_identifier.geoqa_concept_identifier import NgramClassIdentifier
from src.engine.class_identifier.description_based_identifiers import SparseClassIdentifier, DenseClassIdentifier, HybridClassIdentifier
from src.engine.class_identifier.llm_identifier import LlmClassIdentifier
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier


if __name__ == '__main__':

    # --------------------------------------------
    # ----- Parse the command line arguments -----
    # --------------------------------------------
    AVAILABLE_DATASETS = ['BEASTIARY', 'ELECTIONS', 'GEOQ1089', 'WEBQSP', 'CWQ', 'QALD9', 'QALD10', 'LC-QuAD', 'LC-QuAD-2']
    AVAILABLE_MODELS = ['GEOQA', "GEOQA-R", 'SPARSE', 'DENSE', 'HYBRID-1', 'HYBRID-2', 'HYBRID-3', 'HYBRID-4', 'LLM', 'DEV', 'DEV3']
    
    parser = argparse.ArgumentParser(
        description="Process a list of datasets."
    )
    
    parser.add_argument(
        '--dataset',
        nargs='+',
        required=True,
        choices=AVAILABLE_DATASETS,
        help=f"List of datasets separated by spaces. Possible options: {', '.join(AVAILABLE_DATASETS)}"
    )
    parser.add_argument(
        '--model',
        nargs='+',
        required=True,
        choices=AVAILABLE_MODELS,
        help=f"One or more models to use. Possible options: {', '.join(AVAILABLE_MODELS)}"
    )
    parser.add_argument(
        '--topk',
        nargs='+',
        default='3',
        help=f"List of top-k parameters to try."
    )
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help="Print results while they are being generated."
    )
    parser.add_argument(
        '--output',
        default='./crd_evaluation.md',
        help="Path to the output file. Default is './evaluation.md'."
    )
    
    args = parser.parse_args()
    print("Datasets provided:")
    datasets = []
    for dataset in args.dataset:
        print(f"- {dataset}")
        datasets.append(dataset)
        
    print("Models provided:")
    models = []
    for model in args.model:
        print(f"- {model}")
        models.append(model)
        
    print("Top-k provided:")
    k_values = []
    for k_value in args.topk:
        print(f"- {k_value}")
        k_values.append(int(k_value))
        
    # ----------------------------
    # ----- Evaluation logic -----
    # ----------------------------
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(get_relative_path("config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    evaluations = []
    performance_per_model = {}
    best_in_dataset_f1 = {}
    best_in_dataset_recall = {}
    for dataset in tqdm(datasets, desc="Datasets", leave=True):
        # ----- Prepare dataset -----
        gold_func = None
        if dataset == 'BEASTIARY':
            dataset_config = config['beastiary']
            dataset = BeastiaryDataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'ELECTIONS':
            dataset_config = config['elections']
            dataset = ElectionsConcepts.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'GEOQ1089':
            dataset_config = config['geoquestions1089']
            dataset = Geoquestions1089Dataset.from_files(get_relative_path(dataset_config['path']))
            dataset = dataset.y2geo_subset()
        elif dataset == 'WEBQSP':
            dataset_config = config['webqsp']
            dataset = WebQSPDataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'CWQ':
            dataset_config = config['cwq']
            dataset = CwqDataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'QALD9':
            dataset_config = config['qald9']
            dataset = Qald9Dataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'QALD10':
            dataset_config = config['qald10']
            dataset = Qald10Dataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'LC-QuAD':
            dataset_config = config['lc_quad_1']
            dataset = LcQuad1Dataset.from_files(get_relative_path(dataset_config['path']))
        elif dataset == 'LC-QuAD-2':
            dataset_config = config['lc_quad_2']
            dataset = LcQuad2Dataset.from_files(get_relative_path(dataset_config['path']))
        else:
            raise ValueError(f"Invalid dataset. Expected one of {AVAILABLE_DATASETS}, got {args.dataset}")
        
        if gold_func is None:
            gold_class_identifier =  GoldClassIdentifier(
                knowledge_graph=dataset.get_knowledge_graph(),
                endpoint_url=KnowledgeGraph.get_endpoint(dataset.get_knowledge_graph()),
                prefixes=dataset.get_prefixes()
            )
            gold_func = lambda entry: gold_class_identifier.identify(dataset.get_query(entry))
        
        # ----- Evaluation of models -----
        for model in tqdm(models, desc="Models", leave=False):
            # ----- Prepare model -----
            for description_path in tqdm(dataset_config['class_description_paths'], desc="Descriptions", leave=False):
                with open(get_relative_path("../../engine/class_identifier/" + description_path), 'r') as file:
                    line_count = sum(1 for _ in file)
                    print(f"Number of lines in {description_path}: {line_count}")
                
                for k_value in k_values:
                    if k_value > line_count:
                        k_value = line_count / 2
                    prediction_func = None
                    if model == 'SPARSE':
                        identifier = SparseClassIdentifier(description_path, k_value)
                    elif model == 'DENSE':
                        identifier = DenseClassIdentifier(description_path, k_value)
                    elif model == 'HYBRID-1':
                        identifier = HybridClassIdentifier(description_path, k_value, FUSION_MODES.RECIPROCAL_RANK)
                    elif model == 'HYBRID-2':
                        identifier = HybridClassIdentifier(description_path, k_value, FUSION_MODES.RELATIVE_SCORE)
                    elif model == 'HYBRID-3':
                        identifier = HybridClassIdentifier(description_path, k_value, FUSION_MODES.DIST_BASED_SCORE)
                    elif model == 'HYBRID-4':
                        identifier = HybridClassIdentifier(description_path, k_value, FUSION_MODES.SIMPLE)
                    elif model == 'LLM':
                        identifier = LlmClassIdentifier(description_path)
                    elif model == 'DEV':
                        identifier = MixClassIdentifier(description_path, k_value)
                    elif model == 'DEV3':
                        identifier = MixClassIdentifier3(description_path, model_id=SupportedLLMs.VLLM, top_k=k_value)
                    elif model == 'GEOQA':
                        identifier = NgramClassIdentifier(description_path)
                        prediction_func = identifier.api_identify
                    elif model == 'GEOQA-R':
                        if not description_path.endswith('_classes.txt') and not description_path.endswith('_classes_only.txt'):
                            continue
                        identifier = NgramClassIdentifier(description_path)
                    else:
                        raise ValueError(f"Invalid dataset. Expected one of {AVAILABLE_MODELS}, got {args.model}")
                
                    # ----- Evaluate model -----
                    evaluator = Evaluator(task="crd",
                                          model=identifier, 
                                          dataset=dataset, 
                                          gold_func=gold_func,
                                          prediction_func=prediction_func)
                    evaluator.evaluate(logging=True, log_dir=get_relative_path(f"./logs/{timestamp_str}/"))
                    
                    # ----- Bookeeping -----
                    evaluator_log = evaluator.get_metrics()
                    evaluations.append(evaluator_log)
                    
                    if dataset.get_name() not in best_in_dataset_f1:
                        best_in_dataset_f1[dataset.get_name()] = evaluator_log
                    else:
                        if evaluator_log.get_metrics()['f1'] > best_in_dataset_f1[dataset.get_name()].get_metrics()['f1']:
                            best_in_dataset_f1[dataset.get_name()] = evaluator_log
                            
                    if dataset.get_name() not in best_in_dataset_recall:
                        best_in_dataset_recall[dataset.get_name()] = evaluator_log
                    else:
                        if evaluator_log.get_metrics()['recall'] > best_in_dataset_recall[dataset.get_name()].get_metrics()['recall']:
                            best_in_dataset_recall[dataset.get_name()] = evaluator_log
                            
                    if identifier.get_name() not in performance_per_model:
                        performance_per_model[identifier.get_name()] = [evaluator_log.get_metrics()]
                    else:
                        performance_per_model[identifier.get_name()].append(evaluator_log.get_metrics())

                    # ----- Free memory -----
                    del identifier
                    del evaluator
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # ----- Print metrics -----
                    if args.info:
                        print(evaluator_log.get_metrics())
            
    # -----------------------
    # ----- Log results -----
    # -----------------------
    with open(args.output, 'w') as f:
        # Write the table header
        f.write("## Average Model Performance (Average F1)\n")
        f.write("| Model | Precision | Recall | F1 Score |\n")
        f.write("|-------|-----------|--------|----------|\n")

        for model, metrics_list in performance_per_model.items():
            total_f1 = 0
            total_precision = 0
            total_recall = 0
            for m in metrics_list:
                total_f1 += m["f1"]
                total_precision += m["precision"]
                total_recall += m["recall"]
            avg_f1 = total_f1 / len(metrics_list)
            avg_precision = total_precision / len(metrics_list)
            avg_recall = total_recall / len(metrics_list)
                
            f.write(
                "| `{}` | `{:.4f}` | `{:.4f}` | `{:.4f}` |\n".format(
                    model,
                    avg_precision,
                    avg_recall,
                    avg_f1,
                )
            )
        
        f.write("## Best in Dataset (F1)\n")
        f.write("| Dataset | Model | Resource | Precision | Recall | F1 Score | TP Samples | Queries Correct |\n")
        f.write("|---------|-------|----------|-----------|--------|----------|------------|-----------------|\n")

        for dataset, evaluator in best_in_dataset_f1.items():
            m = evaluator.get_metrics()
            f.write(
                "| `{}` | `{}` | `{}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{}` | `{}` / `{}` |\n".format(
                    dataset,
                    evaluator.model,
                    evaluator.resource,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["gold_tp"],
                    m["queries_correct"],
                    evaluator.dataset_length
                )
            )
            
        f.write("## Best in Dataset (Recall)\n")
        f.write("| Dataset | Model | Resource | Precision | Recall | F1 Score | TP Samples | Queries Correct |\n")
        f.write("|---------|-------|----------|-----------|--------|----------|------------|-----------------|\n")

        for dataset, evaluator in best_in_dataset_recall.items():
            m = evaluator.get_metrics()
            f.write(
                "| `{}` | `{}` | `{}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{}` | `{}` / `{}` |\n".format(
                    dataset,
                    evaluator.model,
                    evaluator.resource,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["gold_tp"],
                    m["queries_correct"],
                    evaluator.dataset_length
                )
            )
            
        f.write("## Complete Results\n")
        f.write("| Dataset | Model | Resource | Precision | Recall | F1 Score | Queries Correct |\n")
        f.write("|---------|-------|----------|-----------|--------|----------|-----------------|\n")

        for evaluator in evaluations:
            m = evaluator.get_metrics()
            f.write(
                "| `{}` | `{}` | `{}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{}` / `{}` |\n".format(
                    evaluator.dataset,
                    evaluator.model,
                    evaluator.resource,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["queries_correct"],
                    evaluator.dataset_length
                )
            )