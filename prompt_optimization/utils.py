"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import json
import config
import string
import numpy as np


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split("\n"):
        line = line.strip()

        if line.startswith("# "):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(
                str.maketrans("", "", string.punctuation)
            )
            result[current_header] = ""
        elif current_header is not None:
            result[current_header] += line + "\n"

    return result


def chatgpt(
    prompt,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=10240,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    timeout=30,
):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "openai/gpt-oss-20b",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "reasoning_effort": "low"
    }
    retries = 0
    while True:
        try:
            r = requests.post(
                f"{config.BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=timeout,
            )
            if r.status_code != 200:
                print("FAILED HERE: ", r.json())
                retries += 1
                time.sleep(5)
                if retries > 10:
                    raise Exception(
                        f"API request failed with status {r.status_code}: {r.text}"
                    )
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
            if retries > 10:
                raise Exception("API request timeout after 10 retries")
    r = r.json()
    # print(r)
    if "choices" not in r or not r["choices"]:
        raise Exception(f"Invalid API response: {r}")
    results = []
    for choice in r["choices"]:
        if "message" in choice and "content" in choice["message"]:
            content = choice["message"]["content"]
            if content is not None:
                results.append(content)
            else:
                results.append("")
        else:
            results.append("")
    return results if results else [""]

def chatgpt_with_confidence(
    prompt,
    temperature=0.5,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=10240,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    timeout=30,
):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        # "reasoning_effort": "low",
        "logprobs": "true",
    }
    retries = 0
    while True:
        try:
            r = requests.post(
                f"{config.BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=timeout,
            )
            if r.status_code != 200:
                print("FAILED HERE: ", r.json())
                retries += 1
                time.sleep(5)
                if retries > 10:
                    raise Exception(
                        f"API request failed with status {r.status_code}: {r.text}"
                    )
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
            if retries > 10:
                raise Exception("API request timeout after 10 retries")
    r = r.json()
    if "choices" not in r or not r["choices"]:
        raise Exception(f"Invalid API response: {r}")
    results = []
    confidences = []
    for choice in r["choices"]:
        if "message" in choice and "content" in choice["message"]:
            content = choice["message"]["content"]
            if content is not None:
                results.append(content)
            else:
                results.append("")
        else:
            results.append("")
        if "logprobs" in choice and "content" in choice["logprobs"]:
            logprobs = choice["logprobs"]["content"][0]["logprob"]
            confidence = np.exp(logprobs)
            confidences.append(confidence)
        else:
            confidences.append(0.5)
    print(choice)
    return results, confidences


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "openai/gpt-oss-20b",  # 🔹 rẻ hơn nhiều so với text-davinci-003
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
    }
    while True:
        try:
            r = requests.post(
                f"{config.BASE_URL}/completions",
                headers={
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=10,
            )
            if r.status_code != 200:
                time.sleep(2)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r["choices"]

def format_exemplar(ex):
    """
    Convert a training example dict into a clean exemplar string.

    Supports keys: text/label, input/output, question/answer.
    Maps label 0 -> "No", label 1 -> "Yes".
    """

    # 1. Handle (text, label)
    if "text" in ex and "label" in ex:
        label = ex["label"]
        if label == 0:
            label_str = "No"
        elif label == 1:
            label_str = "Yes"
        else:
            label_str = str(label)

        return f"Text: {ex['text']}\nLabel: {label_str}"
