import re
import time
from typing import List
from lark import Lark, Tree, Token
from src.datasets.dataset import KnowledgeGraph
from src.datasets.geoquestions1089_dataset import Geoquestions1089Dataset
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier
from src.engine.entity_linking.gold_entity_identifier import GoldEntityLinker
from src.metrics import get_kgaqa_tracker
from src.utils import SupportedLLMs, llm_call, is_entity_placeholder, is_uri, is_property_description
from src.logging import LogType, LoggingOptions, create_logger, log, LogComponent, LogLevel
from src.engine.qa.relation_identifier_prompts import _PROMPT_CREATE_GRAPH_REASONING_PATH, _PROMPT_TASK_EXPLANATION, \
    _PROMPT_GRAMMAR_RULES, _PROMPT_GRAMMAR_EXAMPLES, _PROMPT_TASK_EXAMPLES, _PROMPT_REMINDERS, _PROMPT_PREVIOUS_GENERATIONS


_GRAPH_REASONING_PATH_LANG_GRAMMAR = r"""
start: statement

statement: initial_class_or_entity "->" member_only
         | initial_class_or_entity "->" rhs

member_only: "member"

rhs: relation_statement
   | property_description
   | class_or_entity   // Note: member removed from here

relation_statement: class_or_entity "->" rhs

property_description: STRING

initial_class_or_entity: URI

class_or_entity: URI | CAPS

URI: "<" URI_STRING ">"
CAPS: /[A-Z0-9_][A-Z0-9_]*/

URI_STRING: /[^<>"]+/
STRING: "\"" /[^"]*/ "\""

%import common.WS
%ignore WS
"""


class RelationIdentifier():
    def __init__(self, model_id: SupportedLLMs = SupportedLLMs.GPT4_1_MINI, verbalization_model_id: SupportedLLMs = SupportedLLMs.GPT4_1_MINI):
        super().__init__()
        self.model_id = model_id
        self.verbalization_model_id = verbalization_model_id
        self.parser = Lark(_GRAPH_REASONING_PATH_LANG_GRAMMAR, start="start", parser="lalr")
        log(
            f"RelationIdentifier initialized with model {self.model_id} and verbalization model {self.verbalization_model_id}",
            LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.HEADER
        )
        
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------

    def identify(self, sentence: str, classes: List[str], entities: List[str], trials: int = 5) -> List:
        log(f"Identifying relations for: {sentence} with classes: {classes} and entities: {entities}",
            LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
        if (classes == "" and entities == "") or (entities == [] and classes == []) or (not classes and not entities):
            log("No classes or entities given. Cannot identify relations.",
                LogComponent.RELATION_IDENTIFIER, LogLevel.ERROR, LogType.NORMAL)
            return []
        
        start_time = time.time() # Start the timer
        
        generated_connections = [] # List of lists that contains the connections that form the generated graph reasoning paths.
        generated_reasoning_paths = [] # Contains reasoning paths, as strings where every connection is separated by a new line.
        
        connections = []
        while not connections and trials > 0:
            get_kgaqa_tracker()._ri_total_trials += 1
            trials -= 1
            
            if classes == None or not classes:
                classes = []
            if entities == None or not entities:
                entities = []
                
            # Clean up any None entries
            classes = [str(c) for c in classes if c is not None]
            entities = [str(e) for e in entities if e is not None]
                        
            # Generate graph reasoning path
            prompt, generated = self.generate_graph_reasoning_path(
                sentence=sentence,
                classes=", ".join(classes),
                entities=", ".join(entities),
                previous_generations=generated_reasoning_paths
            )
            
            # Extract the connections that form the generated graph reasoning path
            connections = self.extract_graph_reasoning_path_connections_from_generation(generated)   
            
            # If no connections were generated, try again
            if len(connections) == 0:
                continue
            
            # Save the generated reasoning path for future reference
            generated_connections.append(connections)
            generated_reasoning_paths.append("\n".join(connections))
            
            # Check the grammatical and semantic correctness of the generated reasoning path
            for c in connections:
                # grammar check
                if self.grammar_check(c, classes + entities) == False:
                    log(f"Detected an invalid use of the grammar or hallucination: {c}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
                    connections = []  # Reset connections to try again
                    break
                # semantic check through verbalization of the graph reasoning path
                verbalization = self.verbalize_reasoning_path(c) # we say reasoning path, but it is actually a connection (i.e., a partial graph reasoning path)
                verbalization_is_valid = self.check_verbalization(prompt, generated, c, verbalization)
                if not verbalization_is_valid:
                    connections = []  # Reset connections to try again
                    break
                
        get_kgaqa_tracker()._ri_total_time += time.time() - start_time # Stop the timer and add the time taken to the total time
                
        # a valid reasoning path has been generated
        if connections:
            get_kgaqa_tracker()._ri_valid_generations += 1
            log(f"Valid graph reasoning path generated: {connections}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.HEADER)
            return connections
        
        # reasoning paths were generated, but none of them were valid
        if not connections and generated_connections:
            get_kgaqa_tracker()._ri_invalid_generations += 1
            log(
                "No valid graph reasoning path could be generated after multiple attempts. Selecting the first path...",
                LogComponent.RELATION_IDENTIFIER, LogLevel.WARNING, LogType.NORMAL
            )
            return generated_connections[0]  # Return the first generated reasoning path if no valid connections were found
        
        # no reasoning paths were generated at all
        get_kgaqa_tracker()._ri_no_generations += 1
        log("No graph reasoning path could be generated.", LogComponent.RELATION_IDENTIFIER, LogLevel.ERROR, LogType.NORMAL)
        return []
    
    # -----------------------------------------
    # ----- Generate Graph Reasoning Path -----
    # -----------------------------------------
    
    def generate_graph_reasoning_path(self, sentence: str, classes: str, entities: str, previous_generations: List[str]) -> list:
        # prepare the prompt
        prompt = _PROMPT_CREATE_GRAPH_REASONING_PATH.format(
            sentence=sentence,
            classes=classes,
            entities=entities,
            task_explanation=_PROMPT_TASK_EXPLANATION,
            grammar_rules=_PROMPT_GRAMMAR_RULES,
            grammar_examples=_PROMPT_GRAMMAR_EXAMPLES,
            task_examples=_PROMPT_TASK_EXAMPLES,
            reminders=_PROMPT_REMINDERS,
            previous_generations=_PROMPT_PREVIOUS_GENERATIONS.format(previous_paths="\n\n-----\n\n".join(previous_generations), grammar=_GRAPH_REASONING_PATH_LANG_GRAMMAR) if previous_generations else "")
        log(f"Generated prompt for graph reasoning path: {prompt}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.PROMPT)
        # generate
        get_kgaqa_tracker()._ri_prompt_llm_call += 1
        start_time = time.time()  # Start the timer for LLM call
        generated = llm_call(self.model_id, prompt, max_tokens=1024, temperature=0.2*len(previous_generations)) # temperature is inversely proportional to the number of previous generations, so that the model does not get stuck in a loop of generating the same reasoning path.
        get_kgaqa_tracker()._ri_prompt_llm_time += time.time() - start_time  # Stop the timer and add the time taken to the total time
        log(f"Generated graph reasoning path: {generated}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.LLM_RESULT)
        return prompt, generated
        
    def extract_graph_reasoning_path_connections_from_generation(self, generated: str) -> list:
        # Identify the part of the generated text that contains the graph reasoning path connections.
        search_for_answer_string = generated
        if "FINAL ANSWER" in generated:
            search_for_answer_string = generated.split("FINAL ANSWER")[-1]
        # Extract the connections from the search_for_answer_string
        pat = r'(?<=\{).+?(?=\})'
        connections = re.findall(pat, search_for_answer_string)
        connections = list(set(connections))
        return connections
    
    # ------------------------------------------------
    # ----- Semantic Check through Verbalization -----
    # ------------------------------------------------
    
    def verbalize_reasoning_path(self, reasoning_path):
        hops = reasoning_path.split(" -> ")
        hops = [hop.strip() for hop in hops]
        verbalization = []
        for idx in range(len(hops) - 1):
            start = hops[idx]
            end = hops[idx + 1]
            if is_uri(end) or is_entity_placeholder(end):
                verbalization.append(f"get the paths that connect {start} to {end}")
            elif "member" in end:
                verbalization.append(f"get all the members of {start}")
            elif is_property_description(end):
                verbalization.append(f"get the {end} property of {start}")
        return ", then ".join(verbalization)
    
    def check_verbalization(self, prompt: str, generated: str, reasoning_path: str, verbalization: str):
        verbalization_prompt = prompt + "\n" + generated + "\n The verbalization of the partial graph reasoning path\n" + reasoning_path + "\nis\n" + verbalization + "\n Does the verbalization match the path? The verbalization is meant to help you understand when you have made a mistake in the order of connections. The verbalization is not incorrect. It captures the connections of the path fully correctly, but it does not explain the actual meaning of each connection. Therefore if the verbalization is 'misleading' or 'incorrect' or 'slightly wrong' then the path is wrong and should be classified as incorrect. Explain your reasoning, but put the final answer as a yes or no in curly braces { }\n"
        log(f"Verbalization prompt: {verbalization_prompt}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.PROMPT)
        get_kgaqa_tracker()._ri_prompt_verbalization_call += 1
        start_time = time.time()  # Start the timer for LLM call
        verbalization_generation = llm_call(self.verbalization_model_id, verbalization_prompt, 512)
        log(f"Verbalization generation: {verbalization_generation}", LogComponent.RELATION_IDENTIFIER, LogLevel.DEBUG, LogType.LLM_RESULT)
        # Check if the verbalization is valid
        get_kgaqa_tracker()._ri_prompt_verbalization_time += time.time() - start_time  # Stop the timer and add the time taken to the total time
        if r"{no}" in verbalization_generation.lower():
            log(f"Invalid verbalization for reasoning path: {reasoning_path} - {verbalization}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
            return False
        else:
            log(f"Valid verbalization for reasoning path: {reasoning_path} - {verbalization}", LogComponent.RELATION_IDENTIFIER, LogLevel.INFO, LogType.NORMAL)
            return True
    
    # -------------------------
    # ----- Grammar Check -----
    # -------------------------    
    
    def extract_uris(self, tree):
        """Recursively extract all URIs from the parse tree."""
        uris = []

        if isinstance(tree, Tree):
            if tree.data == "URI":
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

            # Check if all URIs are in the allowed list
            if allowed_uris is not None and allowed_uris:
                for uri in uris:
                    if uri not in allowed_uris:
                        return False

            return True
        except Exception:
            return False


if __name__ == '__main__':
    create_logger("relation_identifier", "LOGGER_pATH", log_option=LoggingOptions.LOG_TO_CONSOLE, log_level=LogLevel.DEBUG)
    
    dataset = Geoquestions1089Dataset.from_files('PATH_TO_FILE')
    
    identifier = RelationIdentifier(SupportedLLMs.GPT4_1_MINI)
    
    kg = dataset.get_knowledge_graph()
    
    entity_linker = GoldEntityLinker(knowledge_graph=kg, prefixes=dataset.get_prefixes())
    class_identifier = GoldClassIdentifier(knowledge_graph=kg, endpoint_url=KnowledgeGraph.get_endpoint(kg), prefixes=dataset.get_prefixes())
    
    for idx in [218, 275, 276]:
        entry = dataset[idx]
        print(entry)
        
        question = dataset.get_question(entry)
        gold_query = dataset.get_query(entry)
        
        classes = class_identifier.identify(gold_query)
        entities = entity_linker.identify(gold_query)
        entities = [uri for uri in entities if uri not in classes]
        
        connections = identifier.identify(question, classes, entities)
        print(f"Connections: {connections}\n")
        
        entry['connections'] = connections
        dataset[idx] = entry