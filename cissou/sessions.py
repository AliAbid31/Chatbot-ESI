import time
from collections import OrderedDict
from threading import Lock
from typing import List, Dict

class UserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.last_active = time.time()

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.last_active = time.time()

class SessionManager:
    def __init__(self, max_sessions: int = 500):
        self.sessions: OrderedDict[str, UserSession] = OrderedDict()
        self.lock = Lock()
        self.max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> UserSession:
        with self.lock:
            if session_id in self.sessions:
                self.sessions.move_to_end(session_id)
                return self.sessions[session_id]
            
            if len(self.sessions) >= self.max_sessions:
                self.sessions.popitem(last=False)  # Nettoyage LRU

            new_session = UserSession(session_id)
            self.sessions[session_id] = new_session
            return new_session