import openai
import sys
import time
import logging
import os # For API key
# import numpy as np # np.power can be replaced with math.power or base ** exponent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORTANT ---
# Set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'
# or uncomment and set it directly below.
# openai.api_key = "YOUR_OPENAI_API_KEY" 
# It's more secure to use environment variables.
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    logging.warning("OPENAI_API_KEY environment variable not set. API calls may fail.")
    # You might want to raise an error here or handle it as per your application's needs
    # For example: raise ValueError("OPENAI_API_KEY not found.")

def call_openai_chat_api(
    prompt: str,
    model_name: str = "gpt-3.5-turbo",
    max_tokens: int = 512,
    temperature: float = 0.7,
    # logprobs parameter for ChatCompletion returns top_logprobs if True,
    # it doesn't take an integer for number of logprobs like Completion.
    # To get logprobs, you must use a model that supports it and set logprobs=True.
    # The response will include logprob information if available.
    request_logprobs: bool = False,
    # The 'echo' parameter is not directly supported in Chat Completions.
    # The API returns the full chat history including the user's prompt.
    # If you need to mimic the old 'echo' behavior, you'd typically prepend
    # the prompt to the assistant's reply yourself if echo was True.
    # For simplicity, this parameter is removed, as echo=False was the default.
    verbose: bool = False
):
    """
    Calls the OpenAI Chat Completion API with the given prompt and parameters.
    Retries on rate limit errors with exponential backoff.

    Args:
        prompt (str): The user's prompt.
        model_name (str): The model to use (e.g., "gpt-3.5-turbo", "gpt-4").
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        request_logprobs (bool): Whether to request logprobs.
                                 Note: The structure of logprobs in the response
                                 is different from the legacy Completion API.
        verbose (bool): If True, prints more detailed logs.

    Returns:
        openai.types.chat.chat_completion.ChatCompletion: The API response object,
        or None if a non-retryable error occurs.
    """
    response = None
    received = False
    num_api_errors = 0 # Renamed from num_rate_errors for clarity, as it handles more than just rate errors

    # Chat Completions API expects a list of messages
    messages = [{"role": "user", "content": prompt}]

    if verbose:
        logging.info(f"Attempting to call OpenAI API with model: {model_name}, prompt: \"{prompt[:100]}...\"")

    while not received:
        try:
            # Create the OpenAI client if it doesn't exist or if the API key might have changed
            # This is good practice if the key could be updated dynamically.
            # For most simple scripts, initializing once globally is fine.
            client = openai.OpenAI() # Uses API key from env or openai.api_key

            completion_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if request_logprobs:
                completion_params["logprobs"] = True
                # If you need to specify top_logprobs count (e.g. for gpt-3.5-turbo-0125+)
                # completion_params["top_logprobs"] = 5 # Or your desired number

            response = client.chat.completions.create(**completion_params)
            received = True
            if verbose:
                logging.info("API call successful.")

        except openai.APIConnectionError as e:
            num_api_errors += 1
            logging.error(f"OpenAI API request failed to connect: {e} (Attempt {num_api_errors})")
            time.sleep(2 ** num_api_errors) # Exponential backoff

        except openai.RateLimitError as e:
            num_api_errors += 1
            logging.error(f"OpenAI API request exceeded rate limit: {e} (Attempt {num_api_errors})")
            # Consider a max retry limit
            if num_api_errors > 5: # Example max retries
                logging.critical("Exceeded maximum retry attempts for rate limit error.")
                return None
            time.sleep(2 ** num_api_errors) # Exponential backoff

        except openai.APIStatusError as e: # General API error (e.g. 500, 503)
            num_api_errors += 1
            logging.error(f"OpenAI API returned an API Status Error: {e.status_code} - {e.response} (Attempt {num_api_errors})")
            if num_api_errors > 5:
                logging.critical("Exceeded maximum retry attempts for API status error.")
                return None
            time.sleep(2 ** num_api_errors)

        except openai.BadRequestError as e: # Replaces InvalidRequestError for openai >= 1.0
            logging.critical(f"OpenAI API request was invalid: {e.body.get('message') if e.body else e}")
            logging.critical(f"Prompt passed in:\n\n{prompt}\n\n")
            # This is a non-retryable error for this specific request
            return None # Or assert False / raise e if you want to halt execution

        except Exception as e: # Catch any other unexpected errors
            num_api_errors += 1
            logging.error(f"An unexpected error occurred: {e} (Attempt {num_api_errors})")
            if num_api_errors > 5:
                logging.critical("Exceeded maximum retry attempts for unexpected error.")
                return None
            time.sleep(2 ** num_api_errors) # Exponential backoff, use with caution for unknown errors

    return response

if __name__ == '__main__':
    # Example Usage:
    # Make sure your OPENAI_API_KEY is set as an environment variable.
    # For example, in your terminal: export OPENAI_API_KEY='your_key_here'

    if not openai.api_key:
        print("OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    sample_prompt = "Translate the following English text to French: 'Hello, how are you?'"
    print(f"Sending prompt: \"{sample_prompt}\"")

    # Simple call
    api_response = call_openai_chat_api(sample_prompt, verbose=True)

    if api_response:
        # The main content is in choices[0].message.content
        # print("\nFull API Response:")
        # print(api_response)

        print("\nAssistant's Reply:")
        if api_response.choices:
            print(api_response.choices[0].message.content)
        else:
            print("No choices returned in the response.")

        # Example with logprobs (if supported by model and requested)
        # Note: gpt-3.5-turbo (older versions) might not return logprobs by default.
        # Newer versions like gpt-3.5-turbo-0125 do.
        print("\n--- Example with logprobs ---")
        api_response_logprobs = call_openai_chat_api(
            "What is the capital of France?",
            model_name="gpt-3.5-turbo-0125", # Ensure this model supports logprobs
            request_logprobs=True,
            verbose=True
        )
        if api_response_logprobs:
            if api_response_logprobs.choices and api_response_logprobs.choices[0].logprobs:
                print("Logprobs received:")
                # The structure of logprobs is complex.
                # For example, to print logprobs for each token in the response:
                for token_logprob_info in api_response_logprobs.choices[0].logprobs.content:
                    print(f"Token: '{token_logprob_info.token}', Logprob: {token_logprob_info.logprob}")
            else:
                print("Logprobs not found in the response. The model might not support them or they weren't returned.")
            print("\nAssistant's Reply (logprobs example):")
            if api_response_logprobs.choices:
                 print(api_response_logprobs.choices[0].message.content)

    else:
        print("Failed to get a response from the API.")

    # Example of a potentially problematic prompt (e.g., too long if not handled by truncation)
    # long_prompt = "Repeat the word 'test' 10000 times. " # This would likely exceed token limits
    # print("\n--- Example with a potentially problematic prompt ---")
    # response_problem = call_openai_chat_api(long_prompt, max_tokens=50, verbose=True) # Short max_tokens for testing
    # if response_problem:
    #     print(response_problem.choices[0].message.content)
    # else:
    #     print("API call failed as expected or due to other issues for the problematic prompt.")

