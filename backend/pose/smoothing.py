from collections import deque

class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.values = deque(maxlen=window)

    def update(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)
