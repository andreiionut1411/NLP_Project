from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import argparse # For command-line arguments
import os

# Load model directly (globally, so it's loaded once per script execution)
# Ensure the model is downloaded or accessible in the environment where this script runs.
try:
    tokenizer = AutoTokenizer.from_pretrained("krotima1/AlignScoreCS")
    model = AutoModelForSequenceClassification.from_pretrained("krotima1/AlignScoreCS")

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=2 # Get top 2 predictions to check for non-neutral labels
    )
except Exception as e:
    # If model loading fails, print error and allow script to exit
    # The calling ensemble script should handle if this script fails critically.
    print(json.dumps({"error": f"Failed to load AlignScoreCS model: {str(e)}", "score": 0.0}), flush=True)
    clf = None # Indicate model loading failure

def convert_result(label):
    if label == 'LABEL_1':
        return "SUPPORTS"
    elif label == 'LABEL_0':  
        return "REFUTES"
    elif label == 'LABEL_2':
        return "NEUTRAL"
    else:
        return "UNKNOWN"


def compute_alignScorePrediction(text, claim):
    """
    Computes the alignment score prediction for a given text and claim.
    """
    # Assuming the model is already loaded and available as 'clf'
    results = clf({"text": text, "text_pair": claim})
    maxScore = -1
    label = "UNKNOWN"
    #print(f"AlignScoreCS results: {results}")  # Debugging line to see the output
    for result in results:
        if result['score'] > maxScore and result['label'] != "LABEL_2":
            # Let's ignore the NEUTRAL label
            maxScore = result['score']
            label = result['label']
    predicted = convert_result(label)
    return predicted

def main():
    parser = argparse.ArgumentParser(description="AlignScoreCS checker script.")
    parser.add_argument("--claim", required=True, type=str, help="The text (claim) to fact-check.")
    parser.add_argument("--context", required=True, type=str, help="The context to fact-check against the claim.")
    # Optional: Add an argument for context if you want to pass it
    # parser.add_argument("--context", type=str, default="", help="The context for the claim.")
    args = parser.parse_args()

    claim_to_check = args.claim
    claim_context = args.context


    if clf is None: # Model didn't load, error already printed
        # The ensemble script expects a JSON output, so provide one for error.
        # Redundant if model loading error already printed JSON, but good for other main() failures.
        print(json.dumps({"error": "AlignScoreCS model not available.", "score": 0.0}), flush=True)
        return

    try:
        numerical_score = compute_alignScorePrediction(claim_context, claim_to_check)
        print(json.dumps({"score": numerical_score}), flush=True)
    except Exception as e:
        error_message = f"Error during AlignScore processing in main: {str(e)}"
        print(json.dumps({"error": error_message, "score": 0.0}), flush=True)

if __name__ == "__main__":
    main()