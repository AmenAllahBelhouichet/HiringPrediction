import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Chargement du modèle et des encodeurs
model = joblib.load("xgb_model_v5.pkl")

# Fonction pour décoder les valeurs de 'past_applications'
def decode_past_applications(scaled_value):
    if 0.0 <= scaled_value <= 1.0:
        return round(scaled_value * 10)
    else:
        raise ValueError("Value should be between 0.0 and 1.0")

def decode_application_status(encoded_value):
    mapping = {
        0.0: 'Rejected',
        0.5: 'Pending',
        1.0: 'Accepted'
    }
    if encoded_value in mapping:
        return mapping[encoded_value]
    else:
        raise ValueError("Encoded value must be 0.0, 0.5, or 1.0")

def decode_age(encoded_value, min_age=20, max_age=53):
    return round(encoded_value * (max_age - min_age) + min_age)

def decode_experience(encoded_value, min_exp=0, max_exp=22):
    return round(encoded_value * (max_exp - min_exp) + min_exp, 2)

def decode_certifications(encoded_value, min_cert=1, max_cert=4):
    return round(encoded_value * (max_cert - min_cert) + min_cert)

def decode_assessment_score(encoded_value, min_score=40, max_score=100):
    return encoded_value * (max_score - min_score) + min_score

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = request.form
            print("Form Data:", data)

            # Vérification de la présence des données nécessaires
            required_fields = ['age', 'experience', 'certifications', 'assessment_score', 'past_applications', 'application_status']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            # Décoder et préparer les features pour la prédiction
            features = []
            
            # Décodage des données du formulaire en utilisant les fonctions appropriées
            features.append(decode_age(float(data['age'])))  # Décoder l'âge
            features.append(decode_experience(float(data['experience'])))  # Décoder l'expérience
            features.append(decode_certifications(float(data['certifications'])))  # Décoder les certifications
            features.append(decode_assessment_score(float(data['assessment_score'])))  # Décoder le score d'évaluation
            features.append(decode_past_applications(float(data['past_applications'])))  # Décoder les applications passées
            features.append(decode_application_status(float(data['application_status'])))  # Décoder le statut de la candidature

            # Convertir features en format NumPy array 2D pour la prédiction
            features = np.array(features).reshape(1, -1)

            # Faire la prédiction
            prediction = model.predict(features)
            result = "Hired" if prediction[0] == 1 else "Not Hired"

            return render_template('result.html', result=result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        except Exception as e:
            return jsonify({"error": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
