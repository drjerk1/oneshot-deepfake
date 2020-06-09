class QuitSignal(Exception):
    def __init__(self, code):
        self.code = code
    def __int__(self):
        return int(self.code)

class ErrorSignal(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return str(self.message)
