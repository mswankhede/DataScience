from flask import Flask, request, jsonify
import util
app  = Flask(__name__)

##getting the location
@app.route('/get_locations')
def get_locations():
    response = jsonify({
        'locations': util.get_locations()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

## getting the price
@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    util.load_artifacts()
    app.run(debug=True)