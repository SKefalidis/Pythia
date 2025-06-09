class KgaqaTracker:
    def __init__(self):
        # ---------------------------
        # ----- Dataset Metrics -----
        # ---------------------------
        self._total = 0 # queries that were processed and counted
        self._invalid_gold_queries = 0
        self._empty_gold_queries = 0
        # -------------------------------
        # ----- Performance Metrics -----
        # -------------------------------
        # Simple Query Generator
        self._total_tp = 0
        self._total_fp = 0
        self._total_fn = 0
        self._total_macro_f1 = 0
        self._total_macro_precision = 0
        self._total_macro_recall = 0
        self._total_hits_at_1 = 0
        self._exact_match = 0
        # ICL Query Generator
        self._icl_total_tp = 0
        self._icl_total_fp = 0
        self._icl_total_fn = 0
        self._icl_total_macro_f1 = 0
        self._icl_total_macro_precision = 0
        self._icl_total_macro_recall = 0
        self._icl_total_hits_at_1 = 0
        self._icl_exact_match = 0
        # -------------------
        # ----- General -----
        # -------------------
        self._total_questions = 0           # total number of questions (they might not all be counted, because they might have invalid gold queries)
        self._llm_time = 0.0                # total time taken by the LLM to generate answers
        self._llm_calls = 0                 # total number of LLM calls
        self._embed_time = 0.0              # total time taken for embedding generation
        self._embed_calls = 0               # total number of embedding calls
        self._sparql_execs = 0              # total number of SPARQL executions
        self._sparql_time = 0.0             # total time taken for SPARQL executions
        self._ri_time = 0.0                 # total time taken for relation identification
        self._pe_time = 0.0                 # total time taken for path extraction
        self._qg_zero_shot_time = 0.0       # total time taken for query generation
        self._qg_icl_time = 0.0             # total time taken for query generation with in-context learning
        # -------------------------------
        # ----- Relation Identifier -----
        # -------------------------------
        self._ri_valid_generations = 0      # how many questions received valid generations
        self._ri_invalid_generations = 0    # how many questions received invalid generations
        self._ri_no_generations = 0         # how many questions received no generations
        self._ri_total_trials = 0
        self._ri_total_time = 0.0
        self._ri_prompt_llm_call = 0         # how many times the LLM was called for relation identification
        self._ri_prompt_llm_time = 0.0       # how much time was spent on the LLM for relation identification
        self._ri_prompt_verbalization_call = 0  # how many times the LLM was called for verbalization
        self._ri_prompt_verbalization_time = 0.0  # how much time was spent on the LLM for verbalization
        # --------------------------
        # ----- Path Extractor -----
        # --------------------------
        self._pe_no_triples_for_connection = 0      # how many connections had no triples found
        self._pe_no_triples_for_property_path = 0   # how many connections had no triples for property path
        self._pe_known_to_known = 0                 # how many connections were known to known
        self._pe_unknown_goal = 0
        self._pe_unknown_start = 0
        self._pe_shortest_calls = 0                 # how many connections were found with the shortest path
        self._pe_shortest_path_time = 0.0
        self._pe_all_paths_calls = 0                # how many connections were found with all paths
        self._pe_all_paths_time = 0.0
        self._pe_graph_search_calls = 0                # how many connections were found with graph search
        self._pe_graph_search_time = 0.0
        self._pe_neighborhood_calls = 0                # how many connections were found with neighborhood search
        self._pe_neighborhood_time = 0.0
        self._pe_property_path_to_triples_calls = 0
        self._pe_property_path_to_triples_time = 0.0
        self._pe_prompt_inclusion_time = 0.0
        self._pe_prompt_inclusion_calls = 0
        self._pe_prompt_grounding_time = 0.0
        self._pe_prompt_grounding_calls = 0
        self._pe_prompt_neighborhood_time = 0.0
        self._pe_prompt_neighborhood_calls = 0
        # ---------------------------
        # ----- Query Generator -----
        # ---------------------------
        self._qg_prompt_query_gen_zero_shot_calls = 0
        self._qg_prompt_query_gen_zero_shot_time = 0.0
        self._qg_prompt_query_gen_icl_calls = 0
        self._qg_prompt_query_gen_icl_time = 0.0

        
    def get_metrics(self):
        return {
            "top_metrics 0-shot": {
                "f-score": self._total_macro_f1/ self._total if self._total > 0 else 0.0,
                "hits@1": self._total_hits_at_1 / self._total if self._total > 0 else 0.0,
                "accuracy": self._exact_match / self._total if self._total > 0 else 0.0,
                "time_per_question": (self._ri_time + self._pe_time + self._qg_zero_shot_time) / self._total_questions if self._total_questions > 0 else 0.0,
                "average_llm_calls": (self._llm_calls - self._qg_prompt_query_gen_icl_calls) / self._total_questions if self._total_questions > 0 else 0.0,
            },
            "top_metrics ICL": {
                "f-score": self._icl_total_macro_f1 / self._total if self._total > 0 else 0.0,
                "hits@1": self._icl_total_hits_at_1 / self._total if self._total > 0 else 0.0,
                "accuracy": self._icl_exact_match / self._total if self._total > 0 else 0.0,
                "time_per_question": (self._ri_time + self._pe_time + self._qg_icl_time) / self._total_questions if self._total_questions > 0 else 0.0,
                "average_llm_calls": (self._llm_calls - self._qg_prompt_query_gen_zero_shot_calls) / self._total_questions if self._total_questions > 0 else 0.0,
            },
            "dataset_metrics" : {
                "total_questions": self._total_questions,
                "total_valid": self._total,
                "invalid_gold_queries": self._invalid_gold_queries,
                "empty_gold_queries": self._empty_gold_queries
            },
            "performance_metrics": {
                "total_tp": self._total_tp,
                "total_fp": self._total_fp,
                "total_fn": self._total_fn,
                "total_macro_f1": self._total_macro_f1,
                "total_macro_precision": self._total_macro_precision,
                "total_macro_recall": self._total_macro_recall,
                "total_hits_at_1": self._total_hits_at_1,
                "exact_match": self._exact_match,
                "icl_total_tp": self._icl_total_tp,
                "icl_total_fp": self._icl_total_fp,
                "icl_total_fn": self._icl_total_fn,
                "icl_total_macro_f1": self._icl_total_macro_f1,
                "icl_total_macro_precision": self._icl_total_macro_precision,
                "icl_total_macro_recall": self._icl_total_macro_recall,
                "icl_total_hits_at_1": self._icl_total_hits_at_1,
                "icl_exact_match": self._icl_exact_match
            },
            "general_metrics": {
                "llm_time": self._llm_time,
                "llm_calls": self._llm_calls,
                "embed_time": self._embed_time,
                "embed_calls": self._embed_calls,
                "sparql_execs": self._sparql_execs,
                "sparql_time": self._sparql_time,
                "ri_time": self._ri_time,
                "pe_time": self._pe_time,
                "qg_zero_shot_time": self._qg_zero_shot_time,
                "qg_icl_time": self._qg_icl_time
            },
            "relation_identifier_metrics": {
                "ri_valid_generations": self._ri_valid_generations,
                "ri_invalid_generations": self._ri_invalid_generations,
                "ri_no_generations": self._ri_no_generations,
                "ri_total_trials": self._ri_total_trials,
                "ri_total_time": self._ri_total_time,
                "ri_prompt_llm_call": self._ri_prompt_llm_call,
                "ri_prompt_llm_time": self._ri_prompt_llm_time,
                "ri_prompt_verbalization_call": self._ri_prompt_verbalization_call,
                "ri_prompt_verbalization_time": self._ri_prompt_verbalization_time
            },
            "path_extractor_metrics": {
                "pe_no_triples_for_connection": self._pe_no_triples_for_connection,
                "pe_no_triples_for_property_path": self._pe_no_triples_for_property_path,
                "pe_known_to_known": self._pe_known_to_known,
                "pe_unknown_goal": self._pe_unknown_goal,
                "pe_unknown_start": self._pe_unknown_start,
                "pe_shortest_calls": self._pe_shortest_calls,
                "pe_shortest_path_time": self._pe_shortest_path_time,
                "pe_all_paths_calls": self._pe_all_paths_calls,
                "pe_all_paths_time": self._pe_all_paths_time,
                "pe_graph_search_calls": self._pe_graph_search_calls,
                "pe_graph_search_time": self._pe_graph_search_time,
                "pe_neighborhood_calls": self._pe_neighborhood_calls,
                "pe_neighborhood_time": self._pe_neighborhood_time,
                "pe_property_path_to_triples_calls": self._pe_property_path_to_triples_calls,
                "pe_property_path_to_triples_time": self._pe_property_path_to_triples_time,
                "pe_prompt_inclusion_time": self._pe_prompt_inclusion_time,
                "pe_prompt_inclusion_calls": self._pe_prompt_inclusion_calls,
                "pe_prompt_grounding_time": self._pe_prompt_grounding_time,
                "pe_prompt_grounding_calls": self._pe_prompt_grounding_calls,
                "pe_prompt_neighborhood_time": self._pe_prompt_neighborhood_time,
                "pe_prompt_neighborhood_calls": self._pe_prompt_neighborhood_calls
            },
            "query_generator_metrics": {
                "qg_prompt_query_gen_zero_shot_calls": self._qg_prompt_query_gen_zero_shot_calls,
                "qg_prompt_query_gen_zero_shot_time": self._qg_prompt_query_gen_zero_shot_time,
                "qg_prompt_query_gen_icl_calls": self._qg_prompt_query_gen_icl_calls,
                "qg_prompt_query_gen_icl_time": self._qg_prompt_query_gen_icl_time
            }
        }
        
    def load_from_dict(self, metrics_dict):
        for key, value in metrics_dict.items():
            for key2, value2 in value.items():
                if key2 == "total_valid":
                    key2 = "total"
                if hasattr(self, f"_{key2}"):
                    setattr(self, f"_{key2}", value2)
                else:
                    print(f"Warning: Metric '{key2}' not found in KgaqaTracker.")
    
    def print(self):
        metrics = self.get_metrics()
        for category, values in metrics.items():
            print(f"{category}:")
            max_key_length = max(len(key) for key in values.keys())
            for key, value in values.items():
                print(f"\t{key:<{max_key_length}} : {value}")
        print("\n")
        
tracker = None

def get_kgaqa_tracker() -> KgaqaTracker:
    global tracker
    if tracker is None:
        tracker = KgaqaTracker()
    return tracker

def get_kgaqa_tracker_from_dict(metrics_dict: dict) -> KgaqaTracker:
    global tracker
    if tracker is None:
        tracker = KgaqaTracker()
    tracker.load_from_dict(metrics_dict)
    return tracker