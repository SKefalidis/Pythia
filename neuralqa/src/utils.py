from enum import Enum
import os
import inspect
import time
from transformers import pipeline
import torch
import requests
import openai
import faiss
import numpy as np
from packaging import version
import transformers
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import google.genai as genai
from google.genai import types
from groq import Groq
from src.metrics import get_kgaqa_tracker


# -----------------------------
# ----- General utilities -----
# -----------------------------

def get_relative_path(relative_path):
    """
    Get the relative path of the current script.
    """
    caller_frame = inspect.stack()[1]
    caller_filepath = caller_frame.filename

    caller_dir = os.path.dirname(os.path.abspath(caller_filepath))
    data_path = os.path.join(caller_dir, relative_path)
    return data_path

# -------------------------
# ----- LLM utilities -----
# -------------------------

class SupportedLLMs(Enum):
    """
    Supported LLMs for the application.
    """
    GPT4_1_NANO = "gpt-4.1-nano-2025-04-14"
    GPT4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT4_1 = "gpt-4.1-2025-04-14"
    GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-05-20"
    DEEPSEEK_V3 = "deepseek-chat"
    DEEPSEEK_R1 = "deepseek-reasoner"
    GEMMA2_9B = "google/gemma-2-9b-it"
    GEMMA2_27B = "google/gemma-2-27b-it"
    GEMMA3_4B = "google/gemma-3-4b-it"
    GEMMA3_12B = "google/gemma-3-12b-it"
    GEMMA3_27B = "google/gemma-3-27b-it"
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA2_13B = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA2_70B = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
    LLAMA3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
    VLLM = "vllm-server"
    GROQ = "groq"

LLM_PIPELINES = {}

BASE_URL_VLLM = "YOUT_VLLM_SERVER_URL"
            
def vllm_get_available_model():
    """Query the vLLM server for the available model."""
    response = requests.get(f"{BASE_URL_VLLM}/v1/models")
    if response.status_code == 200:
        models = response.json().get("data", [])
        if models:
            return models[0]["id"]
        else:
            raise ValueError("No models found on the vLLM server.")
    else:
        raise RuntimeError(f"Failed to fetch models: {response.status_code} {response.text}")

def llm_call(llm: SupportedLLMs, prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """
    Call the LLM with the given prompt and additional arguments.
    """
    get_kgaqa_tracker()._llm_calls += 1
    start_time = time.time()
    try:
        generated = ""  
        if "gemini" in llm.value:
            client = genai.Client(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your Google API key
            response = client.models.generate_content(
                model=llm.value,
                contents=prompt,
                config=types.GenerateContentConfig(
                    seed=451, # 0451
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                    ]
                ))
            generated = response.text
        elif "groq" in llm.value:
            client = Groq(
                api_key='YOUR_GROQ_API_KEY',  # Replace with your Groq API key
            )
            response = client.chat.completions.create(
            messages=[
                    {"role": "system",
                    "content": "You are a helpful assistant that tries its best to follow the instructions given by the user to generate a satisfying result. Be concise, helpful and try your best to do what the user requests."},
                    {"role": "user",
                    "content": prompt}
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            generated = response.choices[0].message.content
        elif "gpt" in llm.value:
            while True:
                try:
                    client = openai.OpenAI(
                        api_key="YOUR_OPENAI_API_KEY", # Replace with your OpenAI API key
                    )
                    response = client.chat.completions.create(
                        seed=451, # 0451
                        model=llm.value,
                        temperature=temperature,
                        max_completion_tokens=max_tokens*2, # FIXME: 2x the max tokens to avoid truncation
                        messages=[
                            {"role": "system",
                            "content": "You are a helpful assistant that tries its best to follow the instructions given by the user to generate a satisfying result. Be concise, helpful and try your best to do what the user requests."},
                            {"role": "user",
                            "content": prompt}
                        ]
                    )
                    generated = response.choices[0].message.content
                    break
                except openai.RateLimitError as e:
                    print(f"Rate limit exceeded for {llm.value}. Please try again later.")
                    print(f"Error: {e}")
                    time.sleep(10)
        elif "vllm" in llm.value:
            def chat_with_vllm(prompt) -> requests.Response:
                model_name = vllm_get_available_model()
                headers = {
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                    "stream": False
                }

                response = requests.post(f"{BASE_URL_VLLM}/v1/chat/completions", headers=headers, json=data)

                if response.status_code == 200:
                    return response.json()
                else:
                    raise RuntimeError(f"Chat request failed: {response.status_code} {response.text}")
                                
            response = chat_with_vllm(prompt)
            # print(response)
            generated = response['choices'][0]['message']['content']
        elif "gemma-3" in llm.value:
            if llm.value not in LLM_PIPELINES:
                LLM_PIPELINES[llm.value] = pipeline(
                    "text-generation",
                    model=llm.value,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "load_in_4bit": True,
                        "attn_implementation": "flash_attention_2"
                    },
                    device_map="auto",
                )
            llm_pipeline = LLM_PIPELINES[llm.value]
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            outputs = llm_pipeline(
                messages,
                max_new_tokens=max_tokens,
            )
            generated = outputs[0]["generated_text"][-1]["content"].strip()
        elif "gemma-2" in llm.value:
            if llm.value not in LLM_PIPELINES:
                LLM_PIPELINES[llm.value] = pipeline(
                    "text-generation",
                    model=llm.value,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "load_in_4bit": True,
                        "attn_implementation": "flash_attention_2"
                    },
                    device_map="auto",
                )
            llm_pipeline = LLM_PIPELINES[llm.value]
            messages = [
                {"role": "user", "content": prompt},
            ]
            outputs = llm_pipeline(
                messages,
                max_new_tokens=max_tokens,
            )
            generated = outputs[0]["generated_text"][-1]["content"].strip()
        else:
            print(f"LLM not supported: {llm.value}")
            print(type(llm.value))
        get_kgaqa_tracker()._llm_time += time.time() - start_time
        return generated
    except Exception as e:
        get_kgaqa_tracker()._llm_time += time.time() - start_time
        print(f"Error calling LLM: {e}")
        return None
    
# ----------------------
# ----- Embeddings -----
# ----------------------

# Use for BELA environment!   
transformers_version = transformers.__version__
if version.parse(transformers_version) >= version.parse("4.50.0"):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True,
                                       query_instruction="search_query: ",
                                       text_instruction="search_document: ")
else:
    embed_model = None

def embed(text: str, is_query: bool = True):
    get_kgaqa_tracker()._embed_calls += 1
    embed_model._get_text_embedding
    if is_query:
        return embed_model.get_query_embedding(text)
    else:
        return embed_model.get_text_embedding(text)
    
# ----------------------------
# ----- SPARQL Execution -----
# ----------------------------

import time
from SPARQLWrapper import SPARQLWrapper, JSON
import socket
from urllib.error import URLError, HTTPError

def is_server_up(endpoint, test_query="ASK {}"):
    """
    Sends a lightweight test query to check if the SPARQL endpoint is responsive.
    Returns True if the server responds, False otherwise.
    """
    test_sparql = SPARQLWrapper(endpoint)
    test_sparql.setCredentials("user", "PASSWORD")
    test_sparql.setReturnFormat(JSON)
    test_sparql.setQuery(test_query)
    test_sparql.setTimeout(10)  # quick timeout
    try:
        test_sparql.query().convert()
        return True
    except Exception:
        return False

def execute_sparql_query(query, endpoint, max_wait_minutes=3, retry_interval=30):
    get_kgaqa_tracker()._sparql_execs += 1
    start = time.time()

    sparql = SPARQLWrapper(endpoint)
    sparql.setCredentials("user", "PASSWORD")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.setTimeout(180)  # 3 minutes max

    exception = None
    try:
        query_result = sparql.query()
        get_kgaqa_tracker()._sparql_time += time.time() - start
        return query_result
    except (HTTPError, URLError, socket.timeout, socket.error) as e:
        exception = e
        get_kgaqa_tracker()._sparql_time += time.time() - start
        print(f"[WARN] Query failed or timed out: {e}")
    except Exception as e:
        exception = e
        get_kgaqa_tracker()._sparql_time += time.time() - start
        print(f"[ERROR] Unexpected failure: {e}")

    # Retry logic: poll the server every `retry_interval` until it becomes responsive
    print("[INFO] Checking for server recovery...")
    while not is_server_up(endpoint):
        print(f"[WAIT] Server still down... retrying in {retry_interval} seconds")
        time.sleep(retry_interval)

    raise exception  # Re-raise the last exception after retries

# ---------------------------
# ----- FAISS utilities -----
# ---------------------------
    
def load_faiss_index(index_dir):
    print("Loading FAISS index...")
    index = faiss.read_index(index_dir + "/faiss.index")

    print("Loading documents...")
    with open(index_dir + '/docs.txt', "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    
    return index, documents

def search_faiss_index(index, documents, query, k=5, debug=False):
    query_vector = embed_model.get_query_embedding(query)
    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append(documents[idx])
        if debug:
            print(f"[Distance: {dist:.4f}] {documents[idx]}")
    
    return results

# ---------------------------------
# ----- String type utilities -----
# ---------------------------------

def is_uri(s):
    return "http://" in s or "https://" in s

def is_entity_placeholder(s: str):
    return isinstance(s, str) and s.isupper()

def is_property_description(s: str):
    return not is_entity_placeholder(s)

def is_type_predicate(str):
    if "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in str:
        return True
    if "http://www.wikidata.org/prop/direct/P31" in str:
        return True
    return False