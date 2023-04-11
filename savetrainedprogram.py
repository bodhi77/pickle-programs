import pickle
import joblib

# train the model
model = joblib.load('mnist_dataset.pkl') # your model here

# save the trained model as a pkl file
with open('digit_recognition_model.pkl', 'wb') as f:
    pickle.dump(model, f)
