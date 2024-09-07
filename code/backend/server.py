from flask import Flask, request, jsonify
from ..agentic_rag import *
app = Flask(__name__)

@app.route('/interact', methods=['GET', 'POST'])
def interact():
    if request.method == 'GET':
        # Retrieve query parameters from the request
        question = request.args.get('question', None)

        workflow = build_rag_pipeline()
        rag_agents = workflow.compile()
        output = ask(rag_agents, question)
        print(output)

        # Dummy response for demonstration
        if question:
            return jsonify({
                "message": output
            }), 200
        else:
            return jsonify({
                "error": "Missing parameters"
            }), 400

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
