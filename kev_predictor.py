"""Zero-shot predictions from Kev, served locally by `python -m kev.serve` (System One API).

Same state (build_prompt) and same instruction (QUESTIONS[label]) as Jev and Qwen; Kev exposes
the same API as Jev, so the yes/no probability is returned natively by a Noul question.
Before scoring, the served checkpoint, precision and temperature are checked against the
fingerprint, so a misconfigured server cannot silently produce mislabelled predictions.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from typesafe_sdk import Noul, TypeSafeClient

from prompts import QUESTIONS, build_prompt

QUESTION_ID = "wilson_label"
DTYPE_NAMES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def build_kev_predictor(spec, label, max_attempts=3):
    """Return predict_batch(cts) -> list of P(yes), one per CT, in input order."""
    check_kev_server(spec)

    max_workers = spec.get('max_workers', 4)
    max_chars = spec.get('max_state_chars')

    client = TypeSafeClient(api_key="local", base_url=spec["base_url"], model=spec["model"])
    question = {QUESTION_ID: Noul(instructions=QUESTIONS[label])}

    def predict_one(ct):
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.system_one(state=build_prompt(ct, max_chars=max_chars), questions=question)
                probability = float(response.answers[QUESTION_ID].noul)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"Kev returned {probability} for {ct['nctId']}.")
                return probability
            except Exception as error:          # server busy, connection reset, ...
                if attempt == max_attempts:
                    tqdm.write(f"Kev gave up on {ct['nctId']} after {max_attempts} attempts "
                               f"({type(error).__name__}: {error}); left unscored.")
                    return None

                tqdm.write(f"Kev error on {ct['nctId']} ({type(error).__name__}: {error}); "
                           f"retry {attempt}/{max_attempts - 1} in {attempt}s.")
                time.sleep(attempt)

    def predict_batch(cts):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:   # the server batches concurrent requests
            return list(pool.map(predict_one, cts))

    return predict_batch


def check_kev_server(spec):
    """Fail if the running server does not serve the checkpoint/precision/temperature in the fingerprint."""
    url = spec["base_url"].rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            served = json.load(response)["models"][0]
    except OSError as error:
        raise RuntimeError(f"Kev server not reachable at {url}. Start it with "
                           "`python -m kev.serve --run <model>@<revision> --port 8009`.") from error

    expected = spec["fingerprint"]
    expected_run = f"{expected['model']}@{expected['revision']}"
    problems = []
    if served["run"] != expected_run:
        problems.append(f"checkpoint {served['run']} (expected {expected_run})")
    if served["dtype"] != DTYPE_NAMES.get(expected["dtype"], expected["dtype"]):
        problems.append(f"dtype {served['dtype']} (expected {expected['dtype']}; set KEV_DTYPE)")
    if abs(float(served["temperature"]) - float(expected["temperature"])) > 1e-9:
        problems.append(f"temperature {served['temperature']} (expected {expected['temperature']}; "
                        "set KEV_TEMPERATURE)")
    if served["device"] != "cuda":
        problems.append(f"device {served['device']} (expected cuda; start the server with --device cuda)")
    if problems:
        raise RuntimeError("Kev server does not match the fingerprint: " + "; ".join(problems))

    tqdm.write(f"Kev server OK: {served['run']} on {served['device']}, {served['dtype']}, "
               f"temperature {served['temperature']}.")