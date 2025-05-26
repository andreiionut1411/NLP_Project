input_file = "dataset/document_corpus_clean.txt"
lines_per_file = 15000

with open(input_file, 'r', encoding='utf-8') as infile:
    file_count = 1
    lines = []

    for line in infile:
        lines.append(line)
        if len(lines) >= lines_per_file:
            with open(f"chunk_{file_count}.txt", 'w', encoding='utf-8') as outfile:
                outfile.writelines(lines)
            lines = []
            file_count += 1

    # Write any remaining lines
    if lines:
        with open(f"chunk_{file_count}.txt", 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines)