import requests
import json

def init_prompt_generation(prompt, examples):
    instruction = f"""
    You gave me an instruction on a certain task
    and some example inputs with chain-of-thought.
    I read the instruction carefully and wrote an
    output with chain-of-thought for every input
    correctly.Here is the initial instruction:
    {prompt}
    Here are some correct input-output
    pairs which strictly meet all your
    requirements:
    {examples}
    The instruction given contains the following
    parts. Based on the input-output pairs
    provided, give me the final complete
    instruction in English without any explanation:
    ###Task type###
    Task type: This is a <...> task.
    ###Task detailed description###
    Task detailed description: <Task detailed
    description>
    ###Your output must satisfy the following
    format and constraints###
    Output format(type): <Output format or its
    type>
    Output constraints: <constraints on output>
    ###You must follow the reasoning process###
    <add several reasoning steps if it's
    necessary>
    ###Tips###
    <add several useful tips from a professional
    point of view to accomplish this task better>
    """
    messages = [{"role": "user", "content": instruction}]
    payload = {
        "messages": messages,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.7
    }
    r = requests.post(
        f"http://194.228.55.129:38417/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=30,
    )
    r = r.json()
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


promp = """
# Task
Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.

# Output format
Answer Yes or No as labels

# Prediction
Text: {{ text }}
Label:
"""
examples = [{"label": 0, "text": "Statement: The Chicago Bears have had more starting quarterbacks in the last 10 years than the total number of tenured (UW) faculty fired during the last two decades.\nJob title: Wisconsin Assembly speaker\nState: Wisconsin\nParty: republican\nContext: a an online opinion-piece"},
{"label": 1, "text": "Statement: When Mitt Romney was governor of Massachusetts, we didnt just slow the rate of growth of our government, we actually cut it.\nJob title: Former governor\nState: Massachusetts\nParty: republican\nContext: an interview with CBN News"}]

print(init_prompt_generation(promp, examples))