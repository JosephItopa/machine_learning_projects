from flask import Flask, jsonify, request
from app.simpleinterest import simple_interest
 
# instantiate flask object
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_input():
   """
   A function to get request using flask, evluate and return result.
   """
   packet = request.get_json(force=True)
   principal = packet["principal"]
   rate = packet["rate"]
   period = packet["period"]
 
   interest= simple_interest(principal, rate, period)
 
   return jsonify(packet, interest)