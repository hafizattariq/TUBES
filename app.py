from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
stdscl = pickle.load(open('scaler.pkl', 'rb'))

# Define the route for the home page


@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [
        request.form['sex'],
        float(request.form['age']),
        float(request.form['fare']),
        int(request.form['family_size'])
    ]

    # Preprocess the input data
    features[0] = 1 if features[0].lower(
    ) == 'male' else 0  # Convert 'Sex' to binary
    # Use fit_transform instead of transform
    features[1:] = stdscl.fit_transform([features[1:]])[0]

    # Make a prediction
    prediction = model.predict([features])[0]

    # Map prediction to "Survive" or "Not Survive"
    result = "Survive" if prediction == 1 else "Not Survive"

    # Determine the image URL based on the prediction
    if prediction == 1:  # Survive
        image_url = "https://mmo.aiircdn.com/386/6128da1d05e2b.jpg"
    else:  # Not Survive
        image_url = "https://a.files.bbci.co.uk/worldservice/live/assets/images/2015/06/04/150604031655_china_boat_rescue_body_640x360_afp.jpg"

    # Display the prediction, result, and image on the result page
    return render_template('result.html', prediction=result, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
