# Task
Input: You are a zero-shot classifier responsible for identifying whether a particular action or event **led directly to** the outcome mentioned in the text. Your task is to differentiate between **causal relationships** and **correlational ones**. A causal event is one that played a direct role in bringing about the outcome, as opposed to one that simply coincided with it or had an indirect influence. Evaluate whether the action was essential for the outcome to happen, and whether the outcome would have occurred in the absence of that action. Respond with "Yes" if the event directly caused the outcome, and "No" otherwise.

# Output format
Answer ONLY "Yes" or "No".
NO explanations.
No other words.

# Prediction
Text: {{ text }}
Label: