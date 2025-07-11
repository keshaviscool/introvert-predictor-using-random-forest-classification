from flask import Flask, render_template, request
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model_name = "my_model.joblib"
model = joblib.load(model_name)

app = Flask("__init__")

# Index(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
#        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
#        'Post_frequency', 'Personality'],
#       dtype='object')

def hot_encode(a):
    if a =="yes":
        return 1
    return 0

def ___(a):
    if a=="low":
        return 3
    if a=="medium":
        return 7
    if a=="high":
        return 12
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_ = np.array([
            [
            request.form.get("timeAlone"),
            hot_encode(request.form.get("stageFear")), #yes or no
            request.form.get("socialEvents"),
            request.form.get("goingOutside"),
            hot_encode(request.form.get("drained")), #yes or no
            request.form.get("friendsCircle"),
            ___(request.form.get("postFrequency")) #low, med or high
            ]
         ])
        
        pred = int(model.predict(input_)[0])
        if pred == 1:
            result = "Extrovert"
        if pred == 0:
            result = "Introvert"
        
        return render_template("index.html", result=result, pred=pred)

    return render_template("index.html")

app.run(debug=True)