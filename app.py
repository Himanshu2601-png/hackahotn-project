# app.py
from flask import Flask, request, jsonify
from merge import merge_data  # Import the merge_data function

app = Flask(__name__)

@app.route('/merge', methods=['POST'])
def merge():
    # Get JSON data from the request
    data = request.json
    
    # Extract the data to merge
    data1 = data.get('data1', '')
    data2 = data.get('data2', '')
    
    # Call the merge_data function from merge.py
    result = merge_data(data1, data2)
    
    # Return the result as a JSON response
    return jsonify({'merged_data': result})

if __name__ == '__main__':
    app.run(debug=True)
