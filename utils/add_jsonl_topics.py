import json
import argparse # Import the argparse library

def add_topic_to_jsonl(input_filepath, output_filepath):
    """
    Reads a JSONL file, adds a 'topic' field to each JSON object,
    and writes the modified objects to a new JSONL file.

    Args:
        input_filepath (str): The path to the input JSONL file.
        output_filepath (str): The path to the output JSONL file.
    """
    try:
        with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
            for i, line in enumerate(infile):
                try:
                    # Parse the JSON object from the line
                    json_object = json.loads(line.strip())

                    # Add the 'topic' field
                    json_object['title'] = f'topic{i + 1}'  # i is 0-indexed, so add 1 for 1-based row number

                    # Write the modified JSON object to the output file
                    outfile.write(json.dumps(json_object) + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1} in '{input_filepath}': {line.strip()}")
                except Exception as e:
                    print(f"An error occurred while processing line {i+1} from '{input_filepath}': {e}")
        print(f"Processing complete. Output written to '{output_filepath}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except IOError as e:
        print(f"Error: Could not read from '{input_filepath}' or write to '{output_filepath}'. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Reads a JSONL file line by line, adds a 'topic' field "
                    "(e.g., 'topic<row_number>'), and writes to a new JSONL file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file where results will be saved."
    )

    args = parser.parse_args()

    # --- Call the main function with command-line arguments ---
    add_topic_to_jsonl(args.input_file, args.output_file)

    # --- Example of how to run from command line ---
    # First, create a dummy input file (e.g., input.jsonl):
    # {"id": 1, "text": "This is the first line."}
    # {"id": 2, "data": {"value": 10}, "text": "Second line here."}
    # {"id": 3, "text": "A third entry."}
    #
    # Then run:
    # python your_script_name.py input.jsonl output_with_topics.jsonl
    #
    # You can then inspect 'output_with_topics.jsonl' to see the result.