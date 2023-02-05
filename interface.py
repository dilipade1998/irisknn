from flask import Flask , render_template , jsonify ,request
from project_app.utils import iris_prediction
import pickle
import json
import config

app=Flask(__name__)

@app.route("/") #home API

def base():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])

def result():
    SepalLengthCm = request.form["Sepal Length"]
    SepalWidthCm = request.form["Sepal Width"]
    PetalLengthCm = request.form["Petal Length"]
    PetalWidthCm =request.form["Petal Width"]

    x=iris_prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    final=x.predict_species()
    if final == "Iris-virginica":
        return render_template("virginica.html",data=final)
    elif final == "Iris-setosa":
        return render_template("setosa.html",data=final)
    else:
        return render_template("versicolor.html",data=final)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)