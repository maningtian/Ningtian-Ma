import os, sys, re, json
from flask import Flask, request, jsonify
from agentic_rag import *
from stockformer.inference import init_config, init_model


app = Flask(__name__)


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
        print(output)

        packet = {
            'message': output,
            'symbol': None,
            'action': None,
            'forecast': None,
        }
        if err:
            return jsonify(packet)
        
        pattern = r'{\s*"symbol"\s*:\s*"\w*"\s*,\s*"action"\s*:\s*"\w*"\s*,\s*"days":\s*"?\w*"?\s*}'
        match = re.search(pattern, output)
        if match:
            json_string = match.group()
            packet['message'] = output[:output.find(json_string)].strip()
            try:
                pred_json = json.loads(json_string)
                packet['symbol'] = pred_json['symbol']
                packet['action'] = pred_json['action']
                packet['forecast'] = pred_json['days']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

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
    app.run(host='0.0.0.0', port=5000, debug=True)
