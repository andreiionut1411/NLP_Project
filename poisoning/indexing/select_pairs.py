import json
import random

input_file = "dataset/v1.0-simplified_simplified-nq-train.jsonl"  # Replace with your file path
output_file = "ceva_nou2.jsonl"
sample_size = 150  # Number of random samples

def extract_answer(data):
    tokens = data["document_text"].split()
    annotation = data["annotations"][0]

    yes_no = annotation.get("yes_no_answer", "NONE")
    if yes_no in ("YES", "NO"):
        return yes_no

    short_answers = annotation.get("short_answers", [])
    if short_answers:
        short_texts = [" ".join(tokens[ans["start_token"]:ans["end_token"]]) for ans in short_answers]
        return "; ".join(short_texts)

    long_answer = annotation["long_answer"]
    start = long_answer["start_token"]
    end = long_answer["end_token"]
    if start != -1 and end != -1:
        return " ".join(tokens[start:end])

    return None  # Explicitly return None if no answer

def process_random_nq_data_with_answers(input_file, output_file, sample_size):
    valid_qa_pairs = []
    counter = 0

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                counter += 1

                if counter % 1000 == 0:
                    print(counter)

                data = json.loads(line)
                answer = extract_answer(data)

                if answer:  # Only keep if there's a valid answer
                    question = data["question_text"]
                    valid_qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })

            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    # Randomly sample from valid entries
    sampled_qa = random.sample(valid_qa_pairs, min(sample_size, len(valid_qa_pairs)))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for qa in sampled_qa:
            outfile.write(json.dumps(qa) + "\n")

    print(f"Randomly selected and extracted {len(sampled_qa)} answered Q&A pairs to '{output_file}'.")

# Run the script
process_random_nq_data_with_answers(input_file, output_file, sample_size)
