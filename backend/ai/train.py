import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Example training data
# You will grow this during the hackathon
X = []
y = []

# ---- MANUAL LABELING (IMPORTANT) ----
# Correct reps → label = 1
# Incorrect reps → label = 0

# Example dummy seed data (replace with real data from app)
X.append([120, 3.2, 1.0, 2.1, 0])  # good arm raise
y.append(1)

X.append([70, 0.9, 0.3, 0.8, 0])   # bad arm raise
y.append(0)

model = LogisticRegression()
model.fit(np.array(X), np.array(y))

with open("backend/ai/rep_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
