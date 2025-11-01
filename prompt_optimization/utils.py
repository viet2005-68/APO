"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import json
import config
import string


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
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    timeout=10,
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
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
    r = r.json()
    return [choice["message"]["content"] for choice in r["choices"]]


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
