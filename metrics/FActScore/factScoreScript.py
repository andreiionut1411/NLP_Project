from factscore.factscorer import FactScorer
import nltk
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="FactScorer checker script.")
    parser.add_argument("--text", required=True, type=str, help="The text (generation) to fact-check.")
    # You could add more arguments if needed, e.g., for the topic or knowledge source
    parser.add_argument("--topic", type=str, default="General Statement", help="The topic for the generation.")
    # parser.add_argument("--knowledge_source", type=str, default="example", help="Knowledge source to use.")

    args = parser.parse_args()

    generation_to_check = args.text
    # For FactScorer, topics and generations are lists.
    # We'll use a placeholder topic since the calling script only passes one text string.
    # If you have a way to determine the topic, you can implement that here.
    topics_list = [args.topic] # Placeholder topic
    generations_list = [generation_to_check]
    knowledge_source_name = "example" # As per your original script

    try:
        # It's better to download nltk data once if missing, not every run.
        # This check can be done during environment setup.
        # For robustness in a script, you can do it like this:
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            # Corrected package name from 'punkt_tab' to 'punkt'
            # 'punkt_tab' is not a standard NLTK package.
            print("NLTK 'punkt' not found. Downloading...", flush=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            # Handle other potential errors with nltk.data.find, though less common
            print(f"Error checking NLTK data: {e}", flush=True)
            # Fallback or error based on your needs
            # For this example, we'll try to proceed, FactScorer might handle it or fail later

    except Exception as e:
        print(json.dumps({"error": f"NLTK setup error: {str(e)}", "score": 0.0}), flush=True)
        return

    # --- FactScorer Initialization ---
    # Ensure paths are correct and openAIKey.txt is accessible
    # It's more efficient if FactScorer is initialized once (e.g., in an API)
    # but for subprocess, it's re-initialized each time.
    openai_key_file = "/home/avisario/Projects/NLP-SSL/RAGProject/FActScore/openAIKey.txt" # Ensure this path is correct or make it configurable
    data_path = "/home/avisario/Projects/NLP-SSL/RAGProject/jsons/3.5_1_mini_20_topics.jsonl"
    db_path = "/home/avisario/Projects/NLP-SSL/RAGProject/jsons/3.5_1_mini_20_topics.db"

    # Check if OpenAI key file exists
    if not os.path.exists(openai_key_file):
        print(json.dumps({"error": f"OpenAI key file not found: {openai_key_file}", "score": 0.0}), flush=True)
        return
    # Check if data and db paths exist
    if not os.path.exists(data_path):
        print(json.dumps({"error": f"Data path not found: {data_path}", "score": 0.0}), flush=True)
        return
    if not os.path.exists(db_path) and not data_path.endswith(".jsonl"): # .db is created from .jsonl if not exists
        # If db_path is explicitly required and not auto-creatable, add check.
        # FactScorer often creates the .db from .jsonl if db_path doesn't exist.
        pass


    try:
        fs = FactScorer(model_name="npm", openai_key=openai_key_file)
        # The registration seems to return None or self, not to be assigned with a comma.
        fs.register_knowledge_source(
            knowledge_source_name,
            data_path=data_path,
            db_path=db_path
        )

        # Get the score
        # fs.get_score returns a dictionary, e.g.,
        # {'score': 0.xx, 'num_facts_per_response': [y], 'respond_ratio': z.z, ...}
        # or a list of such dictionaries if multiple inputs.
        # For a single generation, it should be a dictionary.
        out = fs.get_score(topics_list, generations_list, knowledge_source=knowledge_source_name)

        # Ensure 'out' is a dictionary and contains the 'score' key
        if isinstance(out, dict) and "init_score" in out:
            final_score = out["init_score"]
            print(json.dumps({"score": final_score}), flush=True)
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "init_score" in out[0]:
            # If it returns a list of results for the list of generations
            final_score = out[0]["init_score"]
            print(json.dumps({"score": final_score}), flush=True)
        else:
            # Fallback if the structure is unexpected
            print(json.dumps({"error": "Score not found in FactScorer output", "score": 0.0, "details": str(out)}), flush=True)

    except Exception as e:
        # Catch any exception during FactScorer usage and return a JSON error
        error_message = f"Error during fact scoring: {str(e)}"
        print(json.dumps({"error": error_message, "score": 0.0}), flush=True) # Default score on error

if __name__ == '__main__':
    main()