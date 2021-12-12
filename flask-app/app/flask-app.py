# import modules
from flask import Flask, jsonify, request
from simpleInterest import simple_interest

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_input():
    """
    A simple flask app to take input & invoke simple interest module to process the input parameters.
    """
    packet = request.get_json(force=True)
    principal = packet["principal"]
    rate = packet["rate"]
    period = packet["period"]

    interest= simple_interest(principal, rate, period)

    return jsonify(packet, interest)

# main driver function
if __name__ == '__main__':
    app.run()