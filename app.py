from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# 🔥 FIX CACHE ISSUE (IMPORTANT)
@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sleep = float(request.form["sleep"])
        gpa = float(request.form["gpa"])
        stress = float(request.form["stress"])
        anxiety = float(request.form["anxiety"])
        steps = float(request.form["steps"])

        data = np.array([[sleep, gpa, stress, anxiety, steps]])
        prediction = model.predict(data)[0]

        if prediction == 0:
            result = "Healthy Mental State"
        elif prediction == 1:
            result = "Mild Risk (Needs Attention)"
        else:
            result = "High Risk (Immediate Support Needed)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)