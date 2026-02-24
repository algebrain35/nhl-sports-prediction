from flask import Blueprint, request, jsonify
from flask_cors import CORS, cross_origin
from flask_login import login_user, logout_user, login_required, current_user
import sys

sys.path.insert(1, './backend/db')
from models import User, db

auth_bp = Blueprint('auth', __name__)

# enabled CORS for auth routes
CORS(auth_bp, supports_credentials=True)

@auth_bp.route('/api/register', methods=['POST'])
@cross_origin(supports_credentials=True)  # allows requests to include credentials (like cookies, session tokens) when communicating with the backend
def register():
    try:
        data = request.get_json()
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        email = data.get('email')
        password = data.get('password')

        if not first_name or not last_name or not email or not password:
            return jsonify({"error": "All fields are required."}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already exists."}), 400

        new_user = User(first_name=first_name, last_name=last_name, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "Registration successful. Please log in."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/login', methods=['POST'])
@cross_origin(supports_credentials=True)  # allows requests to include credentials (like cookies, session tokens) when communicating with the backend
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return jsonify({
                "message": "Logged in successfully!",
                "user": {
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "is_subscribed": user.is_subscribed
                }
            }), 200
        else:
            return jsonify({"error": "Incorrect email or password."}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/logout', methods=['POST'])  # changed to POST request (wasn't anything before)
@cross_origin(supports_credentials=True)  # allows requests to include credentials (like cookies, session tokens) when communicating with the backend
@login_required
def logout():
    try:
        logout_user()
        return jsonify({"message": "Logged out successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/user', methods=['GET'])
@cross_origin(supports_credentials=True)  # allows requests to include credentials (like cookies, session tokens) when communicating with the backend
@login_required
def get_user():
    try:
        return jsonify({
            "first_name": current_user.first_name,
            "last_name": current_user.last_name,
            "email": current_user.email,
            "is_subscribed": current_user.is_subscribed
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
