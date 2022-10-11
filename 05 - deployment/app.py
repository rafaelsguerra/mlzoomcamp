from copyreg import pickle
from flask import Flask, jsonify
from flask import request
import pickle

with open("./model1.bin", "rb") as f:
    model = pickle.load(f)

with open("./dv.bin", "rb") as f:
    dv = pickle.load(f)

app = Flask("prediction_service")


@app.route("/")
def hello():
    return "hello world"


@app.route("/predict", methods=["POST"])
def predict() -> float:
    """Predicts the probability the client will receive the credit card.

    Returns:
        float: probability of receiving a credit card.
    """
    data = request.get_json()
    X = dv.transform([data])
    result = model.predict_proba(X)[0, 1]
    return jsonify({"probability": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
