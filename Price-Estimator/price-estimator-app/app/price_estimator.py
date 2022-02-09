"""
input parameter: 
        'cargoTonnage': , 'Month': ,
        'totalDistance': , 'asset_type_Covered': ,
        'asset_type_Flatbed': , 'asset_type_Open': ,
        'asset_type_Tipper': , 'asset_size_3': ,
        'asset_size_5': , 'asset_size_10': ,
        'asset_size_15': , 'asset_size_20': , 
        'asset_size_28': , 'asset_size_30': ,
        'asset_size_35', 'asset_size_40': , 
        'asset_size_45': , 'asset_size_50': ,
        'asset_size_55': , 'asset_size_60': , 
        'goodType_ANIMAL FEED': , 'goodType_BEVERAGES AND DRINKS': ,
        'goodType_CLINKA': , 'goodType_CORN FLOUR': , 'goodType_DETERGENT': ,
        'goodType_DRINKS': ,     'goodType_EQUIPMENTS': , 'goodType_FISH FEED': ,
        'goodType_FLOUR': , 'goodType_FMCG': , 'goodType_NOODLES': , 'goodType_OIL AND GAS': ,
        'goodType_OTHERS': , 'goodType_PASTA': , 'goodType_POULTRY': , 'goodType_RAW MATERIALS': ,
        'goodType_SAUSAGE': , 'goodType_SESAME SEED': ,'goodType_SORGHUM': , 'goodType_SOYA MILLS': ,
        'goodType_SOYABEAN SEEDS': , 'goodType_SUGAR': , 'goodType_VEDAN': , 'goodType_VEGETABLE OIL': ,
        'goodType_WINE AND SPIRIT': , 'source_LAGOS': , 'source_OGUN': ,     'source_CROSS RIVER': , 'source_KANO': ,
        'source_OYO': , 'source_KADUNA': , 'source_RIVERS': , 'source_KWARA': , 'source_ANAMBRA': , 'source_ADAMAWA': ,
        'source_GOMBE': , 'destination_LAGOS': , 'destination_OGUN': ,  'destination_KANO': , 'destination_OYO': ,
        'destination_GOMBE': ,  'destination_RIVERS': , 'destination_KADUNA': , 'destination_OSUN': ,
        'destination_FEDERAL CAPITAL TERRITORY': , 'destination_BORNO': , 'destination_ANAMBRA': ,
        'destination_EDO': , 'destination_DELTA': ,  'destination_KWARA': , 'destination_PLATEAU': , 'destination_ABIA': ,
        'destination_ONDO': , 'destination_SOKOTO': , 'destination_YOBE': ,  'destination_ADAMAWA': , 'destination_ENUGU: '

output parameter: 
        'predicted amount'
"""

from flask import Flask, jsonify, request
import numpy as np
import joblib
 
# instantiate flask object
app = Flask(__name__)

# load the regression model from disk
filename = 'model/gbr_model.pkl'
loaded_model = joblib.load(open(filename, 'rb'))

changes = 20000

@app.route('/', methods = ['GET', 'POST'])
def get_input():
   """
   A flask script to interface between ml model and the user request.
   """
   # load packets
   price_packet = request.get_json(force = True)

   # extract and transform the input values
   input_data = list(price_packet.values())

   # reshape the dataset
   data = np.array(input_data).reshape(1, 77)

   # generate prediction
   get_result = loaded_model.predict(data)[0]

   result_min = get_result - changes

   result_max = get_result + changes
 
   return jsonify(price_packet, {"predicted_price": get_result, "minimum price": result_min, "maximum price": result_max})