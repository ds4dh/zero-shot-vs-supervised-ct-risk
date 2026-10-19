"""Prompt shared by all zero-shot models (Jev, Kev, Qwen).

Any change to this file changes the predictions: bump PROMPT_VERSION so cached
predictions are not silently reused.
"""
import math

PROMPT_VERSION = "v1"

DOSING_ERROR_INSTRUCTION = (
    "This is a clinical trial registration, written before the trial started. "
    "Predict whether the completed trial will report dosing errors. "
    "Dosing errors are medication errors reported as adverse events: overdose, underdose, "
    "wrong dose, wrong drug, incorrect route or administration, or dosing schedule errors. "
    "Formally, the answer is yes if the lower bound of the 95% Wilson score confidence interval "
    "of the dosing-error rate (reported dosing-error events divided by participants at risk, "
    "summed over all arms) exceeds 0.01%. About 1 in 20 completed trials meet this criterion."
)

QUESTIONS = {"wilson_label": DOSING_ERROR_INSTRUCTION}  

QWEN_SYSTEM_PROMPT = (
    "You are an expert in clinical trial design and medication safety. "
    "Answer the question about the clinical trial with a single word: yes or no."
)


def build_qwen_messages(ct, label):
    user = (
        f"Question: {QUESTIONS[label]}\n\n"
        f"Clinical trial registration:\n{build_prompt(ct)}\n\n"
        "Answer the question above with yes or no.\n"
        "Answer:"
    )
    return [{"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


EXCLUDED_FIELDS = {
    "labels",
    "metadata",
    "nctId",
}


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (str, list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return " ".join(str(value).split())          # collapse newlines and repeated spaces


def _flatten(record, prefix=""):
    """Yield (field_name, text) pairs; nested dicts become 'parent.child'."""
    for key, value in record.items():
        name = f"{prefix}{key}"
        if key in EXCLUDED_FIELDS or _is_missing(value):
            continue
        if isinstance(value, dict):
            yield from _flatten(value, prefix=f"{name}.")
        elif isinstance(value, (list, tuple)):
            items = [_format_value(v) if not isinstance(v, dict) else
                     ", ".join(f"{k}={_format_value(x)}" for k, x in v.items() if not _is_missing(x))
                     for v in value if not _is_missing(v)]
            if items:
                yield name, "; ".join(items)
        else:
            yield name, _format_value(value)

TRUNCATION_MARKER = "\n[... truncated]"

def build_prompt(ct, max_chars=None):
    """Serialise a CT registration as 'field: value' lines, in the dataset's field order."""
    text = "\n".join(f"{name}: {text}" for name, text in _flatten(ct))
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
    return text