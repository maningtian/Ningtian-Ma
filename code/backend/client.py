import requests
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Ask a question.")
    parser.add_argument(
        "-q",
        "--question",
        required=True,
        type=str,
        help="The question to be answered"
    )
    
    args = parser.parse_args()
    
    # Define the API endpoint  
    chat_url = 'http://127.0.0.1:5000/chat'

    # Define query parameters
    params = {
        'question' : args.question
    }

    # Make a GET request with the parameters
    response = requests.get(chat_url, params=params)

    # Print the response
    if response.status_code == 200:
        print("GET Success:", response.json())
    else:
        print("GET Failed:", response.status_code)


    # # Make a POST request with the data
    # response = requests.post(chat_url, json=params)

    # # Print the response
    # if response.status_code == 200:
    #     print("POST Success:", response.json())
    # else:
    #     print("POST Failed:", response.status_code)

        