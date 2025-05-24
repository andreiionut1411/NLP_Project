import json
import random

def extract_statements(jsonl_file, output_file, num_statements=120):
    selected_statements = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            for evidence_group in data["evidence"]:
                for evidence in evidence_group:
                    if len(evidence) >= 4:
                        evidence_page = evidence[2]

                        if not evidence_page:
                            continue

                        if evidence_page[0] in "C":  # Check if it starts with C, D, E, or F
                            selected_statements.append(line.strip())
                            break  # Stop checking once one valid evidence is found
                if len(selected_statements) >= num_statements:
                    break
            if len(selected_statements) >= num_statements:
                break

    selected_statements = list(set(selected_statements))  # Remove duplicates

    # Ensure we only take the requested number of statements
    # selected_statements = random.sample(selected_statements, min(num_statements, len(selected_statements)))

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(selected_statements))

    print(f"Extracted {len(selected_statements)} statements and saved to {output_file}")

# Example usage
extract_statements("paper_test.jsonl", "filtered_statements_good.jsonl")