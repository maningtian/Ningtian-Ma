from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/interact', methods=['GET', 'POST'])
def interact():
    if request.method == 'GET':
        # Retrieve query parameters from the request
        symbol = request.args.get('symbol', None)
        prediction_length = request.args.get('prediction_length', None)

        # Dummy response for demonstration
        if symbol and prediction_length:
            return jsonify({
                "message": f"Received GET request with symbol: {symbol} and prediction_length: {prediction_length}. Returning Inference Response..."
            }), 200
        else:
            return jsonify({
                "error": "Missing parameters"
            }), 400

    elif request.method == 'POST':
        # Retrieve JSON body parameters from the request
        data = request.get_json()
        symbol = data.get('symbol') if data else None
        prediction_length = data.get('prediction_length') if data else None

        # Dummy response for demonstration
        if symbol and prediction_length:
            return jsonify({
                "message": f"Received POST request with symbol: {symbol} and prediction_length: {prediction_length}"
            }), 200
        else:
            return jsonify({
                "error": "Missing parameters"
            }), 400

@app.route('/sendInvestorPersonality', methods=['POST'])
def interact():
    
    symbol = request.args.get('symbol', None)
    prediction_length = request.args.get('prediction_length', None)

    # Dummy response for demonstration
    if symbol and prediction_length:
        return jsonify({
            "message": f"Received GET request with symbol: {symbol} and prediction_length: {prediction_length}. Returning Inference Response..."
        }), 200
    else:
        return jsonify({
            "error": "Missing parameters"
        }), 400
    
if __name__ == '__main__':
    app.run(debug=True)
