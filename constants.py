from datetime import datetime
import os

DATASET_NAME = "CT-DOSING-ERRORS"
DATASET_VERSION = "0.1.0"       


DATASETS_ROOT = os.environ["DATASETS_ROOT"] 

END_POINT_HF_DATASET_PATH = os.path.join(DATASETS_ROOT, DATASET_NAME, DATASET_VERSION)

# Results are stored next to the scripts, in a "Results" folder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")

# folder where each subsampling are stored
SUBSETS_DIR = os.path.join(RESULTS_DIR, "subsets")



# One entry per zero-shot model: how to reach it, and what defines its predictions.
ZERO_SHOT_MODELS = {
    "kev-0.8b": {
        "backend": "system_one",
        "base_url": "http://127.0.0.1:8009", # uses a local version
        "model": "kev-latest",
        "fallback": "train_prevalence",
        "max_workers": 8,
        "api_key_env": None,
        'max_state_chars': 24000,
        "fingerprint": {"model": "jaredpalmer/kev-0.8b", "revision": "v1.0",
                        "dtype": "bf16", "temperature": 1.0, 'max_state_chars': 24000},
    },
    "kev-9b": {
        "backend": "system_one",
        "base_url": "http://127.0.0.1:8009",  
        "model": "kev-latest",
        "fallback": "train_prevalence",
        "max_workers": 1,
        "api_key_env": None,
        "max_state_chars": 24000, 
        "fingerprint": {"model": "jaredpalmer/kev-9b", "revision": "v1.0",
                        "dtype": "bf16", "temperature": 1.0, 'max_state_chars': 24000},
    },
    "jev": {
        "backend": "system_one",
        "base_url": "https://api.typesafe.ai",   
        "model": "jev-1.13.0", 
        "fallback": "train_prevalence",                  
        "api_key_env": "TYPESAFE_API_KEY",
        "fingerprint": {"model": "jev", "version": "1.13.0"},
    },
    "qwen3.5-397b": {
        "backend": "qwen_logprobs",
        "base_url": 'https://ds4dh.unige.ch/vllm/v1',  # vLLM server on Baobab
        "model": "qwen3.5-397b",            
        "api_key_env": "DS4DH_VLLM_API_KEY",
        "fingerprint": {"model": "Qwen/Qwen3.5-397B",
                        "thinking": False, "top_logprobs": 20},
    },
}

