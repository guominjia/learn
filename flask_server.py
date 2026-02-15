from flask import Flask, request, Response, jsonify
from functools import wraps
import json, time, uuid

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    allowed_origin = "*"
    response.headers['Access-Control-Allow-Origin'] = allowed_origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# Updated authentication decorator
def check_auth(username, password):
    return True # Accept any username/password for testing

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)

        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                'Unauthorized', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

@app.route('/route1', methods=['POST'])
@requires_auth
def route1():
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'Missing id'}), 400

    # Prepare the response chunks
    chunks = [
        {
            "type": "Step 1",
            "metadata": {"itemIndex": 0},
            "msg": {"id": str(uuid.uuid4()), "type": "tool", "content": "begin"}
        },
        {
            "type": "Step 2",
            "metadata": {"itemIndex": 2},
            "msg": {"id": str(uuid.uuid4()), "type": "tool", "content": "work in progress"}
        },
        {
            "type": "Step 3",
            "metadata": {"itemIndex": 3},
            "msg": {"id": str(uuid.uuid4()), "type": "tool", "content": "end"}
        }
    ]

    def generate():
        for chunk in chunks:
            data = json.dumps(chunk)
            yield f"data: {data}\n\n"
            time.sleep(2)  # Simulate delay between chunks

    return Response(
        generate(),
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=True)