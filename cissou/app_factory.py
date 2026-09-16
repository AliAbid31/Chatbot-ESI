from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cissou.service import ChatService
from cissou.config import settings

def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(settings.BASE_DIR / "static"))
    CORS(app)
    
    chat_service = ChatService()

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/chat", methods=["POST"])
    def chat_endpoint():
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "default_user")

        if not message:
            return jsonify({"error": "Message empty"}), 400

        result = chat_service.answer_question(session_id, message)
        return jsonify(result)

    return app