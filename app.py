"""
SpamShield - Premium Spam Detection Platform
Delivering clean communication streams for teams at scale
"""

from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import predict, customdata
import os
import logging
import traceback
import secrets
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['JSON_SORT_KEYS'] = False

# Initialize predictor
predictor = predict()


@app.context_processor
def inject_globals():
    """Inject shared template variables."""
    return {'current_year': datetime.utcnow().year}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    """
    Handle spam prediction request
    
    Returns:
        JSON response with prediction and confidence
    """
    try:
        # Get message from request
        if request.is_json:
            data = request.get_json()
            message_text = data.get('message', '')
        else:
            message_text = request.form.get('message', '')
        
        # Validate input
        if not message_text or message_text.strip() == '':
            return jsonify({
                'error': True,
                'message': 'Please enter a message to analyze'
            }), 400

        logging.info(f"Received prediction request for message: {message_text[:50]}...")

        # Create custom data object
        custom_data = customdata(message_text)
        
        # Convert to DataFrame
        data_df = custom_data.data_frame()
        
        # Get prediction
        prediction, confidence = predictor.get_predict(data_df)
        
        # Prepare response
        response = {
            'error': False,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'message': message_text,
            'is_spam': prediction == 'Spam'
        }

        logging.info(f"Prediction: {prediction} ({confidence*100:.2f}% confidence)")

        return jsonify(response)
        
    except ValueError as ve:
        logging.error(f"Validation error: {str(ve)}")
        return jsonify({
            'error': True,
            'message': str(ve)
        }), 400
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        tb = traceback.format_exc()
        logging.error(tb)

        # In production we return a generic message. When debugging is enabled
        # via environment variable DEBUG_PREDICT_ERRORS (truthy values: 1,true,yes),
        # include the exception message and traceback in the JSON response to
        # make diagnosing Render deployment errors easier.
        generic_msg = 'An error occurred during prediction. Please try again.'
        debug_val = os.environ.get('DEBUG_PREDICT_ERRORS', '')
        debug_truthy = str(debug_val).lower() in ('1', 'true', 'yes', 'on')

        if debug_truthy:
            payload = {
                'error': True,
                'message': str(e),
                'traceback': tb
            }
            return jsonify(payload), 500

        return jsonify({
            'error': True,
            'message': generic_msg
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Try to load model
        predictor.load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'fallback': getattr(predictor, 'fallback', False)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/features')
def features():
    """Render features page"""
    return render_template('features.html')

@app.route('/privacy')
def privacy():
    """Render privacy policy page"""
    return render_template('privacy.html')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logging.error(f"Internal error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("SPAMSHIELD - PREMIUM MESSAGE SAFETY SUITE")
    print("=" * 80)
    print("Real-time screening and notification")
    print("Privacy-first handling of message content")
    print("Operational in milliseconds")
    print("=" * 80)
    print("\nServer running at: http://127.0.0.1:5000")
    print("Premium Edition - Professional Grade Protection")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run Flask app (use_reloader=False fixes Windows compatibility issue)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
