
"""Zero-shot predictions from Qwen3.5-397B served by vLLM (OpenAI-compatible API).

P(yes) is read from the first generated token: thinking disabled, greedy decoding,
top-k log-probabilities, normalised as P(yes) / (P(yes) + P(no)).
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

from prompts import build_qwen_messages

YES_TOKENS = {"yes"}
NO_TOKENS = {"no"}


def build_qwen_predictor(spec, label, max_workers=32):
    """Return predict_batch(cts) -> list of P(yes), one per CT, in input order."""
    api_key = os.environ[spec["api_key_env"]] if spec["api_key_env"] else "EMPTY"
    client = OpenAI(base_url=spec["base_url"], api_key=api_key)
    top_logprobs = spec["fingerprint"]["top_logprobs"]

    def predict_one(ct):
        response = client.chat.completions.create(
            model=spec["model"],
            messages=build_qwen_messages(ct, label),
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        candidates = response.choices[0].logprobs.content[0].top_logprobs
        return yes_probability(candidates, ct_id=ct["nctId"])

    def predict_batch(cts):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:  # vLLM batches concurrent requests
            return list(pool.map(predict_one, cts))

    return predict_batch


def yes_probability(candidates, ct_id=None):
    """Normalised P(yes) from the top-k first-token candidates.

    Variants (' Yes', 'YES', ...) are summed. If neither yes nor no appears in the top-k,
    returns 0.5 and logs the CT: count these and report them in the paper.
    """
    p_yes = p_no = 0.0
    for candidate in candidates:
        token = candidate.token.strip().lower()
        if token in YES_TOKENS:
            p_yes += math.exp(candidate.logprob)
        elif token in NO_TOKENS:
            p_no += math.exp(candidate.logprob)

    if p_yes + p_no == 0.0:
        tqdm.write(f"WARNING: no yes/no token in top-{len(candidates)} for {ct_id}; P(yes)=0.5.")
        return 0.5
    return p_yes / (p_yes + p_no)


