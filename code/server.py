import os, sys
from flask import Flask, request, jsonify
from agentic_rag import *
import json

# from code.stockformer.inference import init_config, init_model


app = Flask(__name__)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        # Retrieve query parameters from the request
        question = request.args.get('question', None)
        if not question:
            return jsonify({
                "message": "Parameter [question] is None"
            }), 400

        workflow = build_rag_pipeline()
        rag_agents = workflow.compile()
        output = ask(rag_agents, question)
        print(output)
        # some function to convert output --> response, Optional[forecast]
        # Need to fix what we search for. since itts not outputtin the "JSON" shit
        current_string = output
        if current_string.find("JSON:"):
            result = current_string[current_string.find("JSON"):]
        elif current_string.find("JSON"):
            result = current_string[current_string.find("JSON"):]
        else:
            print("not found")
        result = result[6:]
        print("result:",result)

        data = json.loads(result)

        # parameters
        stockName = data.get('symbol')
        action = data.get('decision')
        days = data.get('days')
        print(stockName,action,days)


        response = output
        return jsonify({
            "message": response,
            "symbol": stockName,
            "decision": action,
            "forecast": days,
        })

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
    app.run(debug=True)
