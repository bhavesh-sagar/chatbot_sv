class SessionManager:
    def __init__(self):
        self.s = {}

    def save(self, key, value):
        self.s[key] = value

    def get(self, key):
        return self.s.get(key)

    def exists(self, key):
        return key in self.s
