import json
import hashlib
import os

DB_FILE = "users_db.json"

def _load_users():
    if not os.path.exists(DB_FILE): return {}
    try:
        with open(DB_FILE, "r") as f: return json.load(f)
    except json.JSONDecodeError: return {}

def _save_users(users_dict):
    with open(DB_FILE, "w") as f: json.dump(users_dict, f, indent=4)

def init_db():
    if not os.path.exists(DB_FILE):
        users = {"admin": make_hash("1234")}
        _save_users(users)

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_credentials(username, password):
    users = _load_users()
    if username in users:
        if users[username] == make_hash(password):
            return True
    return False

def add_user(username, password):
    users = _load_users()
    if username in users: return False
    users[username] = make_hash(password)
    _save_users(users)
    return True