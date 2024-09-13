import requests
import argparse
# from linkpreview import link_preview

if __name__ == "__main__":
    # url = "https://coincodex.com/stock/NVDA/price-prediction/"
    # preview = link_preview(url)
    # print("title:", preview.title)
    # print("description:", preview.description)
    # print("image:", preview.image)
    # print("force_title:", preview.force_title)
    # print("absolute_image:", preview.absolute_image)
    # print("site_name:", preview.site_name)
    # print("favicon:", preview.favicon)
    # print("absolute_favicon:", preview.absolute_favicon)
    
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
    chat_url = 'http://localhost:5000/api/chat'

    # Define query parameters
    params = {
        'question' : args.question
    }

    # Make a GET request with the parameters
    response = requests.get(chat_url, params=params)

    # Print the response
    if response.status_code == 200:
        print('GET Success.')
        packet = response.json()
        print(packet['message'])
        print(packet['symbol'])
        print(packet['action'])
        print(packet['forecast'])
        print(packet['urls'])
    else:
        print(f'GET Failed. Status Code: {response.status_code}')
        print(response.json()['message'])


    # # Make a POST request with the data
    # response = requests.post(chat_url, json=params)

    # # Print the response
    # if response.status_code == 200:
    #     print("POST Success:", response.json())
    # else:
    #     print("POST Failed:", response.status_code)