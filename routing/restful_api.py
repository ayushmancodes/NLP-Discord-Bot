import os
import importlib.util
import numpy as np
from flask import Flask, request, jsonify
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the spam classifier module
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../spam_classifier_model/main.py'))
spec = importlib.util.spec_from_file_location("spam_module", module_path)
spam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spam_module)

# Import the toxic comment classifier module
module_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../toxic_comment_classifier/src/predictor.py'))
spec2 = importlib.util.spec_from_file_location("toxic_predictor", module_path2)
toxic_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(toxic_module)

app = Flask(__name__)

# Routing For Checking Spam
@app.route('/spam', methods=['POST'])
def compute_spam():
    data = request.get_json()  
    input_value = data.get('input_value')  
    if input_value is None:
        return jsonify({"error": "Please provide an 'input_value'."}), 400    
    result = spam_module.predict_spam(input_value) 
    if isinstance(result, np.ndarray):
        result = result.tolist() 
    return jsonify({"result": result})  


# Routing for toxic comment classifier
@app.route('/toxic', methods=['POST'])
def compute_toxic():
    predictor = toxic_module.Predictor()
    data = request.get_json()  
    input_value = data.get('input_value')  
    if input_value is None:
        return jsonify({"error": "Please provide an 'input_value'."}), 400    
    result = predictor.predict(input_value) 
    return jsonify({"result": str(result)})  
       
if __name__ == '__main__':
    app.run(debug=True)
