from bs4 import BeautifulSoup
import json
from p_tqdm import p_map
import itertools

def clean_html(text):
    """Removes HTML tags from the provided text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def batched(iterable, batch_size):
    """Yield successive batches of given size from iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def process_jsonl_line(line):
    """Extract the 'document_text' field from JSONL line and clean it."""
    data = json.loads(line)
    document_text = data.get("document_text", "")
    cleaned_text = clean_html(document_text)
    return cleaned_text

input_path = "ceva.jsonl"  # Input path for the JSONL file
output_path = "cleaned_document_corpus.txt"   # Output path for the cleaned text
batch_size = 5000  # Tune this based on your system's RAM

# Open the JSONL file and write cleaned text to output file
with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    total_lines = sum(1 for _ in open(input_path))  # For rough progress
    f_in.seek(0)

    # Process in batches
    for batch in p_map(lambda b: [process_jsonl_line(line) for line in b], batched(f_in, batch_size), total=total_lines // batch_size):
        for cleaned_line in batch:
            f_out.write(cleaned_line + "\n")

print(f"Cleaned text has been saved to {output_path}")
