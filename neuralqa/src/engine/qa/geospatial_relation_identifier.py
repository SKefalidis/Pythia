import re
import time
import json
from tkinter import E
from typing import List
from lark import Lark, Tree, Token
from src.datasets.dataset import KnowledgeGraph
from src.datasets.geoquestions1089_dataset import Geoquestions1089Dataset
from src.engine.class_identifier.dev import GoldClassIdentifier
from src.engine.qa.basic_query_generator import GoldEntityLinker
from src.metrics import get_kgaqa_tracker
from src.utils import SupportedLLMs, llm_call, is_entity_placeholder, is_uri, is_property_description
from src.logging import LogType, LoggingOptions, create_logger, log, LogComponent, LogLevel
from src.engine.qa.geospatial_relation_identifier_prompts import PROMPT_GEOSPATIAL_RELATIONS


_GEOSPATIAL_LANG_GRAMMAR = r"""
?start: statement+

?statement: relation "(" class_or_entity "," class_or_entity ")"

relation: "distance"
        | "contains"
        | "touch"
        | "overlaps"
        | "crosses"
        | "north_of"
        | "south_of"
        | "east_of"
        | "west_of"

class_or_entity: URI | CAPS

URI: "<" URI_STRING ">"
CAPS: /[A-Z0-9_][A-Z0-9_]*/

URI_STRING: /[^<>"]+/
STRING: "\"" /[^"]*/ "\""

%import common.WS
%ignore WS
"""

_GEOSPATIAL_DISTANCE_LANG_GRAMMAR = r"""
?start: statement+

?statement: distance_statement
          | basic_statement

basic_statement: relation_no_distance "(" class_or_entity "," class_or_entity ")"

distance_statement: "distance" "(" class_or_entity "," class_or_entity ")" distance_constraint?

distance_constraint: COMPARISON NUMBER UNIT

relation_no_distance: "contains"
                    | "touch"
                    | "overlaps"
                    | "crosses"
                    | "north_of"
                    | "south_of"
                    | "east_of"
                    | "west_of"

class_or_entity: URI | CAPS

COMPARISON: "<" | ">" | "="
NUMBER: /[0-9]+(\.[0-9]+)?/
UNIT: /[a-zA-Z]+/

URI: "<" URI_STRING ">"
CAPS: /[A-Z0-9_][A-Z0-9_]*/

URI_STRING: /[^<>"]+/
STRING: "\"" /[^"]*/ "\""

%import common.WS
%ignore WS

"""


class GeospatialRelationIdentifier():
    def __init__(self, model_id: SupportedLLMs = SupportedLLMs.GPT4_1_MINI, verbalization_model_id: SupportedLLMs = SupportedLLMs.GPT4_1_MINI):
        super().__init__()
        self.model_id = model_id
        self.verbalization_model_id = verbalization_model_id
        self.parser = Lark(_GEOSPATIAL_LANG_GRAMMAR, start="start", parser="lalr")
        log(
            f"GeospatialRelationIdentifier initialized with model {self.model_id} and verbalization model {self.verbalization_model_id}",
            LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.HEADER
        )
        
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------

    def identify(self, sentence: str, classes: List[str], entities: List[str], trials: int = 5):
        log(f"Identifying geospatial relations for: {sentence} with classes: {classes} and entities: {entities}",
            LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
        if classes == "" and entities == "":
            log("No classes or entities given. Cannot identify geospatial relations.",
                LogComponent.RELATION_IDENTIFIER, LogLevel.ERROR, LogType.NORMAL)
            return [], sentence
        
        start_time = time.time() # Start the timer
        
        generations = [] # Contains reasoning paths, as strings where every connection is separated by a new line.
        
        geospatial_relations = []
        rewritten_question = ""
        while not geospatial_relations and trials > 0:
            get_kgaqa_tracker()._ri_total_trials += 1
            trials -= 1
            
            # Generate graph reasoning path
            prompt, generated = self.generate(
                sentence=sentence,
                classes=", ".join(classes),
                entities=", ".join(entities),
                previous_generations=generations
            )
            
            # Extract the connections that form the generated graph reasoning path
            try:
                geospatial_relations, rewritten_question = self.extract_response(generated)   
            except Exception as e:
                log(f"Error extracting geospatial relations from generated text: {e}", LogComponent.RELATION_IDENTIFIER, LogLevel.ERROR, LogType.NORMAL)
                continue
            
            # Save the generated reasoning path for future reference
            generations.append("\n".join(generated))
            
            # Check the grammatical and semantic correctness of the generated reasoning path
            for c in geospatial_relations:
                # grammar check
                if self.grammar_check(c, classes + entities) == False:
                    log(f"Detected an invalid use of the grammar or hallucination: {c}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
                    geospatial_relations = []  # Reset connections to try again
                    rewritten_question = ""  # Reset rewritten question
                    break
                
        get_kgaqa_tracker()._ri_total_time += time.time() - start_time # Stop the timer and add the time taken to the total time
                
        # a valid reasoning path has been generated
        if geospatial_relations:
            get_kgaqa_tracker()._ri_valid_generations += 1
            log(f"Identified geospatial relations: {geospatial_relations}\nQuestion rewritten: {rewritten_question}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.HEADER)
            return geospatial_relations, rewritten_question
        
        # no reasoning paths were generated at all
        get_kgaqa_tracker()._ri_no_generations += 1
        log("No geospatial relations where identified.", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.HEADER)
        return [], sentence
    
    # -----------------------------------------
    # ----- Generate Graph Reasoning Path -----
    # -----------------------------------------
    
    def generate(self, sentence: str, classes: str, entities: str, previous_generations) -> list:
        # prepare the prompt
        prompt = PROMPT_GEOSPATIAL_RELATIONS.format(
            sentence=sentence,
            classes=classes,
            entities=entities)
        log(f"Generated prompt for geospatial relation identification: {prompt}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.PROMPT)
        # generate
        get_kgaqa_tracker()._ri_prompt_llm_call += 1
        start_time = time.time()  # Start the timer for LLM call
        generated = llm_call(self.model_id, prompt, max_tokens=512, temperature=0.5*len(previous_generations)) # temperature is inversely proportional to the number of previous generations, so that the model does not get stuck in a loop of generating the same reasoning path.
        get_kgaqa_tracker()._ri_prompt_llm_time += time.time() - start_time  # Stop the timer and add the time taken to the total time
        log(f"Generated response path: {generated}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.LLM_RESULT)
        return prompt, generated
        
    def extract_response(self, generated: str):
        # Identify the part of the generated text that contains the graph reasoning path connections.
        search_for_answer_string = generated
        if "# GEOSPATIAL RELATIONS" in generated:
            search_for_relations_string = generated.split("# GEOSPATIAL RELATIONS")[-1]
        else:
            raise ValueError("The generated text does not contain the expected format for geospatial relations.")
        if "# REWRITTEN QUESTION" in search_for_answer_string:
            search_for_question_string = search_for_answer_string.split("# REWRITTEN QUESTION")[-1]
        else:
            raise ValueError("The generated text does not contain the expected format for the rewritten question.")
        # Extract the connections from the search_for_answer_string
        pat = r'(?<=\{).+?(?=\})'
        geospatial_relations = re.findall(pat, search_for_relations_string)
        geospatial_relations = list(set(geospatial_relations))
        rewritten_question: str = search_for_question_string.strip() # Remove any leading or trailing whitespace
        return geospatial_relations, rewritten_question
    
    # -------------------------
    # ----- Grammar Check -----
    # -------------------------    
    
    def extract_uris(self, tree):
        """Recursively extract all URIs from the parse tree."""
        uris = []

        if isinstance(tree, Tree):
            if tree.data == "uri":
                # Expecting a child that is a URI_STRING token
                for child in tree.children:
                    if isinstance(child, Token) and child.type == "URI_STRING":
                        uris.append(f"<{child}>")  # reconstruct the full URI
            else:
                # Recurse into child nodes
                for child in tree.children:
                    uris.extend(self.extract_uris(child))

        return uris
    
    def grammar_check(self, reasoning_path, allowed_uris):
        """
        Check if the reasoning path is valid according to the grammar.
        :param reasoning_path: The reasoning path to check.
        :param allowed_uris: A list of allowed URIs. If None, all URIs are allowed. The URIs should not contain angle brackets.
        """
        try:
            # Parse the input
            tree = self.parser.parse(reasoning_path)

            # Collect all URIs
            uris = self.extract_uris(tree)
            
            print(f"Extracted URIs: {uris}")
            print(f"Allowed URIs: {allowed_uris}")

            # Check if all URIs are in the allowed list
            if allowed_uris is not None and allowed_uris:
                for uri in uris:
                    if uri not in allowed_uris:
                        return False

            return True
        except Exception:
            return False


if __name__ == '__main__':
    create_logger("relation_identifier", "LOGGER_PATH", log_option=LoggingOptions.LOG_TO_CONSOLE, log_level=LogLevel.DEBUG)
    
    dataset = Geoquestions1089Dataset.from_files('PATH_TO_DATASET_FILE')
    
    identifier = GeospatialRelationIdentifier(SupportedLLMs.GPT4_1_MINI)
    
    kg = dataset.get_knowledge_graph()
    
    entity_linker = GoldEntityLinker(knowledge_graph=kg, prefixes=dataset.get_prefixes())
    class_identifier = GoldClassIdentifier(knowledge_graph=kg, endpoint_url=KnowledgeGraph.get_endpoint(kg), prefixes=dataset.get_prefixes())
    
    for idx in range(895):
        entry = dataset[idx]
        print(entry)
        
        question = dataset.get_question(entry)
        gold_query = dataset.get_query(entry)
        
        classes = class_identifier.identify(gold_query)
        entities = entity_linker.identify(gold_query)
        entities = [uri for uri in entities if uri not in classes]
        
        relations, new_question = identifier.identify(question, classes, entities)
        print(f"Geospatial Relations: {relations}\nRewritten Question: {new_question}\n")
        
        entry['rewritten_question'] = new_question
        entry['geospatial_relations'] = relations
        dataset[idx] = entry
        
    with open('PATH_TO_OUTPUT', 'w') as f:
        json.dump(dataset.dataset, f, indent=4, ensure_ascii=False)