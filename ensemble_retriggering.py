import subprocess
import json # Assuming your scripts output JSON with a 'score' key
import os
THRESHOLD_FACT_SCORE = 0.6

def get_fact_score(text_to_check, topic):
    # Replace with the path to your Python environment that has FActScore installed
    cuda_env_python = "/home/avisario/anaconda3/envs/fs-env/bin/python"
    script_full_path = "./FActScore/factScoreScript.py"
    script_directory = os.path.dirname(script_full_path)
    script_name = os.path.basename(script_full_path)

    process = subprocess.run(
        [cuda_env_python, script_name, "--text", text_to_check, "--topic", topic], 
        capture_output=True, text=True, check=True,
        cwd=script_directory #
    )
    print(f"FActScore: {process.stdout.strip()}") 
    print(f"FActScore errors: {process.stderr.strip()}")
    return json.loads(process.stdout.strip()).get("score")

def get_align_score(claim, context):
    # Replace with the path to your Python environment that has AlignScore installed
    venv_env_python = "./AlignScoreHF/venvAS2/bin/python"
    script_path = "./AlignScoreHF/alignScoreScript.py"
    process = subprocess.run(
        [venv_env_python, script_path, "--claim", claim, "--context", context],
        capture_output=True, text=True, check=True
    )
    print(f"AlignScore: {process.stdout.strip()}") 
    return json.loads(process.stdout.strip()).get("score")

def jsonl_to_dict(file_path):
    """
    Converts a JSONL file to a dictionary.
    Each 'claim' becomes a key, and its value is a dict with 'context', 'answer', and 'GT'.
    """
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            claim = entry.get("claim")
            if claim:
                data_dict[claim] = {
                    "context": entry.get("context"),
                    "answer": entry.get("answer"),
                    "topic": entry.get("title"),
                    "GT": entry.get("GT")
                }
    return data_dict


data = jsonl_to_dict("./jsons/3.5_1_mini_20_topics.jsonl")

recovered_by_prediction = 0
incorrectly_recovered_by_prediction = 0
total_claims = 0
metrics_align = 0
for claim, content in data.items():
    total_claims +=1
    context = content["context"]
    model_answer = content["answer"]
    ground_truth = content["GT"]
    topic = content["topic"]
    print(f"Processing claim {total_claims}: {claim} with GT:{ground_truth} and model answer: {model_answer}")


    fact_score = get_fact_score(claim, topic)
    align_score = get_align_score(claim, context)
    initial_is_correct = (model_answer == ground_truth)

    should_re_trigger = False


    if (fact_score < THRESHOLD_FACT_SCORE) and (align_score == "REFUTES"):
        metrics_align += 1
        print(f"Metrics align: REFUTES. GT {ground_truth} Model answer: {model_answer}")
        if model_answer!= "REFUTES":
             should_re_trigger = True
    elif (fact_score >= THRESHOLD_FACT_SCORE) and (align_score == "SUPPORTS"):
        metrics_align += 1
        if model_answer != "SUPPORTS":
            should_re_trigger = True
        print(f"Metrics align: SUPPORTS. GT {ground_truth} Model answer: {model_answer}")
    
    print(f"Should re-trigger: {should_re_trigger}")

    if should_re_trigger and model_answer != ground_truth:
                recovered_by_prediction += 1

    if should_re_trigger and model_answer == ground_truth:
                incorrectly_recovered_by_prediction += 1
    


print(f"Recovered by prediction: {recovered_by_prediction / total_claims:.2%}")
print(f"Incorrectly recovered by prediction: {incorrectly_recovered_by_prediction / total_claims:.2%}")
print(f"Percentage of metrics reaching same conclusion : {metrics_align / total_claims:.2%}")
