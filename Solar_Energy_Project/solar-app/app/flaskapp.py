"""
input parameter: 
       'month', 'day', 'Daily_Temp', 'Daily_Precip', 'Daily_Humidity',
       'Daily_Pressure', 'Daily_WindDir', 'Daily_WindSpeed', 'Daily_DNI',
       'Daily_GHI'
output parameter: 
        'Daily_radiation'
"""

from flask import Flask, jsonify, request
import numpy as np
import joblib
 
# instantiate flask object
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def get_input():
   """
   A flask script to interface between ml model and the user request.
   """
   # load packets
   packet = request.get_json(force = True)
   # extract and transform the input values
   input_data = list(packet.values())
   # reshape the dataset
   data = np.array(input_data).reshape(1, 10)
   # load the model from disk
   filename = "app/model_gbr.pkl"
   loaded_model = joblib.load(filename)
   # generate prediction
   solar_irradiation = loaded_model.predict(data)[0]
 
   return jsonify(packet, {"solar irradiance":solar_irradiation})