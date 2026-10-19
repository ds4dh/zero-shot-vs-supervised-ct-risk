"""Zero-shot predictions from Jev (TypeSafe System One API).

Same state (build_prompt) and same instruction (QUESTIONS[label]) as Qwen; the yes/no
probability is returned natively by a Noul question, so no answer-format wrapper is needed.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from prompts import QUESTIONS, build_prompt
from typesafe_sdk import Noul, TypeSafeClient
from typesafe_sdk._core.errors import TypeSafeBadRequestError


def build_jev_predictor(spec, label, max_workers=8, max_attempts=5):
    """Return predict_batch(cts) -> list of P(yes), one per CT, in input order."""
    
    client_kwargs = {}
    if spec.get("api_key_env"):
        client_kwargs["api_key"] = os.environ[spec["api_key_env"]]   # KeyError if not set
    if spec.get("base_url"):
        client_kwargs["base_url"] = spec["base_url"]
    if spec.get("model"):
        client_kwargs["model"] = spec["model"]
    client = TypeSafeClient(**client_kwargs)

    question = {'wilson_label': Noul(instructions=QUESTIONS[label])}

    def predict_one(ct):
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.system_one(state=build_prompt(ct), questions=question)
                probability = float(response.answers['wilson_label'].noul)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"Jev returned {probability} for {ct['nctId']}.")
                return probability
            
            except TypeSafeBadRequestError as error:           
                if "max_tokens_exceeded" in str(error):       
                    tqdm.write(f"Jev: {ct['nctId']} exceeds the token limit; left unscored.")
                    return None                               
                raise

            except Exception as error:          # network errors, rate limits, timeouts
                if attempt == max_attempts:
                    tqdm.write(f"Jev failed on {ct['nctId']} after {max_attempts} attempts.")
                    return None
                wait = 2 ** attempt
                tqdm.write(f"Jev error on {ct['nctId']} ({type(error).__name__}: {error}); "
                           f"retry {attempt}/{max_attempts - 1} in {wait}s.")
                time.sleep(wait)

    def predict_batch(cts):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(predict_one, cts))   # preserves input order

    return predict_batch


