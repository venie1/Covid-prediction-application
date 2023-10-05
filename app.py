import io
import pickle
from io import BytesIO
import pandas as pd
from flask import Flask, request, render_template, Response, send_file
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

# Create flask app


flask_app = Flask(__name__)
sarimad = pickle.load(open("sarimad.pkl", "rb"))
prophetd = pickle.load(open("prophetd.pkl", "rb"))
sarimat = pickle.load(open("sarimat.pkl", "rb"))
prophett = pickle.load(open("prophett.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("prediction.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    f = request.files['data_file']
    if not f:
        return "No file"
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    df = pd.read_csv(stream)
    df.drop(columns='geoId', inplace=True)
    df.drop(columns='popData2020', inplace=True)
    df.drop(columns='continentExp', inplace=True)
    df.drop(columns='countryterritoryCode', inplace=True)
    df = df.dropna()
    df['dateRep'] = pd.to_datetime(df['dateRep'], dayfirst=True)
    df['cases'] = df['cases'].abs()
    df['deaths'] = df['deaths'].abs()
    datewise = df.groupby(["dateRep"]).agg({"cases": 'sum', "deaths": 'sum'})
    datewise["Days Since"] = datewise.index - datewise.index.min()
    grouped_country = df.groupby(["countriesAndTerritories", "dateRep"]).agg({"cases": 'sum', "deaths": 'sum'})

    datewise2 = df.groupby(["dateRep"]).agg({"cases": 'sum', "deaths": 'sum'})
    datewise2["Days Since"] = datewise.index - datewise.index.min()
    if request.form['prediction model'] == '1':
        # Forecast

        ts = pd.Series(sarimad.predict(40))
        fig = plt.figure(figsize=(12, 12))
        plt.xlabel('Future days')
        plt.ylabel('Predicted daily cases')
        plt.plot(ts, color='red',
                 label='forecast', linewidth=5)
        plt.ticklabel_format(style='plain', axis='y')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    elif request.form['prediction model'] == '3':
        future = prophetd.make_future_dataframe(periods= 40)
        forecast = prophetd.predict(future)
        fig = prophetd.plot(forecast)
        plt.xlabel('Future days')
        plt.ylabel('Predicted daily cases')
        plt.ticklabel_format(style='plain', axis='y')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    elif request.form['prediction model'] == '2':
        ts = pd.Series(sarimat.predict(40))
        fig = plt.figure(figsize=(12, 12))
        plt.xlabel('Future days')
        plt.ylabel('Predicted daily cases')
        plt.plot(ts, color='red',
                 label='forecast', linewidth=5)
        plt.ticklabel_format(style='plain', axis='y')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    elif request.form['prediction model'] == '4':
        future = prophett.make_future_dataframe(periods=40)
        forecast = prophett.predict(future)
        fig = prophett.plot(forecast, figsize=(12, 12))
        plt.xlabel(' days')
        plt.ylabel('Predicted daily cases')
        plt.ticklabel_format(style='plain', axis='y')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    flask_app.run(debug=True)
