import os, sys, re, json
from dotenv import load_dotenv
from linkpreview import link_preview
from flask import Flask, request, jsonify
from flask_cors import CORS

from agentic_rag import *
from stockformer.inference import init_config, init_model, predict


DEBUG = True
load_dotenv()


app = Flask(__name__)
CORS(app)


def log(msg):
    if DEBUG:
        print(msg)


def get_link_previews(urls):
    links = []
    for url in urls:
        link = {
            'url': url,
            'siteName': None,
            'image': None,
            'title': None,
            'description': None,
        }
        try:
            preview = link_preview(url)
            link['siteName'] = preview.site_name
            link['image'] = preview.absolute_image
            link['title'] = preview.title
            link['description'] = preview.description
        except:
            pass
        links.append(link)
    return links


def packetify(output, packet):
    pattern = r'{\s*"symbols?"\s*:\s*"\w*"\s*,\s*"actions?"\s*:\s*"\w*"\s*,\s*"days?":\s*"?\w*"?\s*}'
    match = re.search(pattern, output)
    if match:
        json_string = match.group()
        packet['message'] = output[:output[:output.find('JSON')].rfind('\n')].strip()

        try:
            pred_json = json.loads(json_string)
            try:
                packet['symbol'] = pred_json['symbol']
            except KeyError:
                packet['symbol'] = pred_json['symbols']
            try:
                packet['action'] = pred_json['action']
            except KeyError:
                packet['action'] = pred_json['actions']
            try:
                packet['forecast'] = pred_json['days']
            except KeyError:
                packet['forecast'] = pred_json['day']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return packet


@app.route('/api/chat', methods=['GET'])
def chat():
    if request.method == 'GET':
        # Retrieve query parameters from the request
        question = request.args.get('question', None)
        if not question:
            return jsonify({
                "message": "Missing parameter [question]"
            }), 400
        
        output, urls, err = ask(rag_agents, question)
        links = get_link_previews(urls)
        log(output)

        packet = {
            'message': output,
            'links': links,
            'symbol': None,
            'action': None,
            'forecast': None,
        }
        if err:
            return jsonify(packet)
        
        packet = packetify(output, packet)
        if packet['symbol'] != "None" and packet['action'] != "None" and isinstance(packet['forecast'], int):
            prediction_length = min([30, 90, 180, 360], key=lambda x: abs(x - packet['forecast']))
            try:
                config = init_config(f"sp500-{prediction_length}d-final")
                model = init_model(config, f"sp500-{prediction_length}d-final")
                packet['forecast'] = predict([packet['symbol']], model, config, prediction_length, min(prediction_length, 30))[0].to_dict()
                packet['forecast']['Num Days'] = prediction_length
            except Exception as err:
                print(err)

        return jsonify(packet), 200


# @app.route('/api/sip', methods=['POST']) # Send Investor Personality = sip
# def sip():
    
#     variably = request.args.get('variably', None)

#     # Dummy response for demonstration
#     if symbol and prediction_length:
#         return jsonify({
#             "message": f"Received GET request with symbol: {symbol} and prediction_length: {prediction_length}. Returning Inference Response..."
#         }), 200
#     else:
#         return jsonify({
#             "error": "Missing parameters"
#         }), 400


if __name__ == '__main__':
    workflow = build_rag_pipeline()
    rag_agents = workflow.compile()
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)
