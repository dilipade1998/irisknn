from flask import Flask , render_template , jsonify ,request
from project_app.utils import iris_prediction
import pickle
import json
import config

app=Flask(__name__)

@app.route("/") #home API

def base():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])

def home():
    SepalLengthCm = request.form["SepalLengthCm"]
    SepalWidthCm = request.form["SepalWidthCm"]
    PetalLengthCm = request.form["PetalLengthCm"]
    PetalWidthCm =request.form["PetalWidthCm"]

    x=iris_prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    final=x.predict_species()

    return render_template("next.html",data=final)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)