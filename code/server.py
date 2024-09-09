import os, sys, re, json
from flask import Flask, request, jsonify
from agentic_rag import *
from stockformer.inference import init_config, init_model, predict


DEBUG = True
app = Flask(__name__)


def log(msg):
    if DEBUG:
        print(msg)


def packetify(output, packet):
    pattern = r'{\s*"symbols?"\s*:\s*"\w*"\s*,\s*"actions?"\s*:\s*"\w*"\s*,\s*"days?":\s*"?\w*"?\s*}'
    match = re.search(pattern, output)
    if match:
        json_string = match.group()
        packet['message'] = output[:output.find(json_string)].strip()
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


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        # Retrieve query parameters from the request
        question = request.args.get('question', None)
        if not question:
            return jsonify({
                "message": "Missing parameter [question]"
            }), 400

        workflow = build_rag_pipeline()
        rag_agents = workflow.compile()
        output, err = ask(rag_agents, question)
        log(output)

        packet = {
            'message': output,
            'symbol': None,
            'action': None,
            'forecast': None,
        }
        if err:
            return jsonify(packet)
        
        packet = packetify(output, packet)
        if packet['symbol'] != "None" and packet['action'] != "None" and isinstance(packet['forecast'], int):
            prediction_length = min([5, 15, 30, 60, 90, 180, 365], key=lambda x: abs(x - packet['forecast']))
            config = init_config(f"sp500-{prediction_length}d")
            model = init_model(config, f"sp500-{prediction_length}d")
            packet['forecast'] = predict([packet['symbol']], model, config)[0].to_dict()

        return jsonify(packet)

    elif request.method == 'POST':
        # Retrieve JSON body parameters from the request
        question = request.args.get('question', None)

        # Dummy response for demonstration
        if question:
            return jsonify({
                "message": f"Received POST request with symbol: {question}"
            }), 200
        else:
            return jsonify({
                "error": "Missing parameters"
            }), 400


# @app.route('/sip', methods=['POST']) # Send Investor Personality = sip
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
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)
