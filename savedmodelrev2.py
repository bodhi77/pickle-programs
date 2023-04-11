import pickle

# assuming the model object is stored in variable `model`
with open('mnist_dataset.pkl', 'wb') as f:
    pickle.dump(model.pkl, f)
