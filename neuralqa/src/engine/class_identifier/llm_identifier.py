from src.engine.class_identifier.class_identifier import ClassIdentifier

import torch
import re
from transformers import pipeline


class LlmClassIdentifier(ClassIdentifier):
    
    IDENTIFICATION_PROMPT = """
    You are an expert knowledge graph user. You are given a natural language question and a list of knowledge graph classes and their descriptions.
    Your job is to identify the classes that are explicitly mentioned in the request. You are replacing a string-similarity tool.
    Do not add classes because you think that they might be needed or relevant. For example, referring to the Lord of the Rings does not mean
    that I want the Book class to be returned. But if the question is about the works of an author, then the Book class is relevant.
    Surround each class with {{ }} and separate them with a newline. Explain your choices.
    
    Question: {question}
    
    Knowledge graph classes with their descriptions:
    {classes_with_descriptions}
    
    Your response:
    """
    
    
    def __init__(self, description_file_path: str, pipe=None):
        super().__init__()
        self.description_file_path = description_file_path
        self.descriptions = open(description_file_path, 'r').read()
        self.model_id = "google/gemma-3-12b-it"
        if pipe is not None:
            self.pipe = pipe
        else:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "load_in_4bit": True,
                    "attn_implementation": "flash_attention_2"
                },
                device_map="auto",
            )

    def identify(self, question: str, top_k: int = -1, threshold: float = 0.0, debug: bool = False):
        messages = [
            {"role": "user", "content": LlmClassIdentifier.IDENTIFICATION_PROMPT.format(
                question=question, classes_with_descriptions=self.descriptions
            )},
        ]
        
        outputs = self.pipe(messages, max_new_tokens=512)
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
        classes = re.findall(r"\{(.*?)\}", assistant_response)
        classes = [i for i in classes]
        
        if debug:
            print(assistant_response)
        
        return list(set(classes))

    def get_name(self):
        return "LLM RETRIEVER"
    
    def get_resource(self):
        return self.description_file_path.split('/')[-1]


if __name__ == '__main__':
    identifier = LlmClassIdentifier()
    for i in range(3):
        classes = identifier.identify(
            question="Where is Lough Ramor located?")
        print(classes)
