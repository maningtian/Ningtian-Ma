import os, sys, re, json
from dotenv import load_dotenv
from linkpreview import link_preview
from flask import Flask, request, jsonify, session
from flask_cors import CORS

from agentic_rag import *
from stockformer.inference import init_config, init_model, predict

DEBUG = True
load_dotenv()

app = Flask(__name__)
app.secret_key = 'thisismysecretkey1102'  
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

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

def get_quiz_questions():
    return [
        {
            "question": "Question 1: Risk Tolerance\nHow do you feel about financial risks?",
            "options": {
                "A": "I avoid risks at all costs.",
                "B": "I am comfortable with moderate risks for better returns.",
                "C": "I embrace risks and actively seek high returns.",
                "D": "I evaluate risks carefully and take only calculated ones."
            }
        },
        {
            "question": "Question 2: Decision-Making Style\nHow do you make investment decisions?",
            "options": {
                "A": "I rely heavily on advice from financial advisors.",
                "B": "I follow trends and what others are doing.",
                "C": "I research thoroughly and trust my analysis.",
                "D": "I make confident decisions based on past success."
            }
        },
        {
            "question": "Question 3: Primary Goal\nWhat is your main investment objective?",
            "options": {
                "A": "Preserving wealth and avoiding losses.",
                "B": "Growing wealth steadily over time.",
                "C": "Achieving significant financial growth quickly.",
                "D": "Building a portfolio that reflects my own research and insights."
            }
        },
        {
            "question": "Question 4: Market Volatility\nHow do you react to market downturns?",
            "options": {
                "A": "I feel anxious and consider selling.",
                "B": "I stay calm and stick to my plan.",
                "C": "I see it as an opportunity to invest more.",
                "D": "I review my strategy and make adjustments as needed."
            }
        },
        {
            "question": "Question 5: Investment Planning Horizon\nHow do you typically view your investments?",
            "options": {
                "A": "I focus on short-term stability and minimizing losses.",
                "B": "I follow what others suggest, as I'm unsure of long-term plans.",
                "C": "I aim for long-term, aggressive growth and high returns.",
                "D": "I develop detailed, personalized strategies for long-term success."
            }
        },
    ]

def determine_investor_personality(responses):
    # Initialize scores for each personality type
    scores = {
        'Preserver': 0,
        'Follower': 0,
        'Accumulator': 0,
        'Individualist': 0
    }

    # Scoring matrix
    scoring = {
        'Preserver': {'A': 3, 'B': 0, 'C': 0, 'D': 1},
        'Follower': {'A': 1, 'B': 3, 'C': 0, 'D': 1},
        'Accumulator': {'A': 0, 'B': 1, 'C': 3, 'D': 2},
        'Individualist': {'A': 0, 'B': 0, 'C': 1, 'D': 3}
    }

    # Apply scoring
    for i, answer in enumerate(responses):
        for personality in scores:
            scores[personality] += scoring[personality][answer]

    # Determine the highest score
    highest_score = max(scores.values())
    # In case of a tie, you may decide how to handle it (e.g., pick the first one)
    investor_personality = [ptype for ptype, score in scores.items() if score == highest_score][0]

    return investor_personality

investor_personality_profiles = {
    'Preserver': (
        "Preservers are highly risk-averse and prioritize financial security over returns. "
        "They focus on preserving their wealth, often opting for conservative investment strategies "
        "such as fixed deposits, bonds, or other low-risk instruments. They are cautious and deliberate, "
        "preferring advice from trusted sources to minimize uncertainty and avoid losses."
    ),
    'Follower': (
        "Followers lack a clear investment strategy and tend to rely on others for advice or follow market trends. "
        "They may not have a strong interest in financial markets and often make decisions based on what is popular. "
        "Followers can benefit from more education and guidance to build a more independent approach to investing."
    ),
    'Accumulator': (
        "Accumulators are confident, ambitious, and willing to take calculated risks for substantial financial growth. "
        "They are proactive in seeking opportunities and have a higher risk tolerance. They actively monitor their investments "
        "and make bold decisions to achieve their financial goals."
    ),
    'Individualist': (
        "Individualists are analytical and self-reliant. They trust their own research and judgment when making investment decisions. "
        "Individualists are methodical and have a strong understanding of financial markets, often developing personalized strategies. "
        "They prefer calculated risks and seek long-term success based on their insights."
    )
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'message': 'No message provided'}), 400
    
    user_message = data['message'].strip().lower()

    # Handle the "Take Quiz" command
    if user_message == "take quiz":
        # Clear the session data
        session.pop('user_context', None)
        session.pop('quiz_state', None)
        session.modified = True

        # Start the quiz by sending the first question
        return handle_quiz('')

    # Check if 'user_context' is stored in the session
    if 'user_context' not in session:
        # Handle the quiz logic
        return handle_quiz(user_message)
    else:
        # Handle the normal chat flow
        return handle_user_question(user_message)

def handle_quiz(user_message):
    # Initialize quiz state in session if not present
    if 'quiz_state' not in session:
        session['quiz_state'] = {
            'current_question': 0,
            'responses': []
        }
    
    quiz_state = session['quiz_state']
    questions = get_quiz_questions()
    
    # If user_message is empty, send the first question
    if not user_message:
        question_data = questions[quiz_state['current_question']]
        session.modified = True  # Mark session as modified
        return jsonify({
            'message': question_data['question'],
            'options': question_data['options']
        })
    else:
        # Process the user's answer
        user_answer = user_message.strip().upper()
        if user_answer not in ['A', 'B', 'C', 'D']:
            # Resend the same question with an error message
            question_data = questions[quiz_state['current_question']]
            return jsonify({
                'message': "Please enter a valid option (A, B, C, or D).\n" + question_data['question'],
                'options': question_data['options']
            })
        
        # Save the response
        quiz_state['responses'].append(user_answer)
        quiz_state['current_question'] += 1
        session['quiz_state'] = quiz_state
        session.modified = True  # Mark session as modified
        
        # Check if there are more questions
        if quiz_state['current_question'] < len(questions):
            # Send the next question
            question_data = questions[quiz_state['current_question']]
            return jsonify({
                'message': question_data['question'],
                'options': question_data['options']
            })
        else:
            # All questions answered, determine personality
            personality = determine_investor_personality(quiz_state['responses'])
            user_context = investor_personality_profiles[personality]
            session['user_context'] = user_context
            session.pop('quiz_state')  # Remove quiz state from session
            session.modified = True  # Mark session as modified
            
            # Inform the user and proceed to handle their question
            return jsonify({
                'message': f"Thank you for completing the quiz! Your investor personality is {personality}. You can now ask your financial questions.",
            })

def handle_user_question(user_message):
    user_context = session['user_context']
    question = user_message
    
    # Prepare inputs including user_context
    inputs = {"question": question, "user_context": user_context}
    
    # Run the RAG agents
    failure_count = 0
    for out in rag_agents.stream(inputs):
        for key, value in out.items():
            if key == 'generate':
                failure_count += 1
            if failure_count > 2:
                return jsonify({
                    'message': "I am sorry. I am having trouble with your request. Please try again.",
                    'links': [],
                    'symbol': None,
                    'action': None,
                    'forecast': None,
                })
    
    output = value['generation']
    urls = value.get('urls', [])
    links = get_link_previews(urls)
    log(output)
    
    packet = {
        'message': output,
        'links': links,
        'symbol': None,
        'action': None,
        'forecast': None,
    }
    
    packet = packetify(output, packet)
    if packet['symbol'] != "None" and packet['action'] != "None" and isinstance(packet['forecast'], int):
        prediction_length = min([30, 90, 180, 360], key=lambda x: abs(x - packet['forecast']))
        try:
            print(f"Attempting forecast on {packet['symbol']}")
            config = init_config(f"sp500-{prediction_length}d-final")
            model = init_model(config, f"sp500-{prediction_length}d-final")
            forecast = predict([packet['symbol']], model, config, prediction_length, min(prediction_length, 30))[0]
            packet['forecast'] = forecast.to_dict()
            packet['forecast']['Num Days'] = prediction_length
            print('Forecast Sucess...')
        except Exception as err:
            print(err)

    return jsonify(packet)

if __name__ == '__main__':
    workflow = build_rag_pipeline()
    rag_agents = workflow.compile()
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)