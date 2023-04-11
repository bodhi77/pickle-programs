import idx2numpy
import pickle

# Load the data from the IDX files
train_images = idx2numpy.convert_from_file('Train Model/train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('Train Model/train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('Test Model/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('Test Model/t10k-labels.idx1-ubyte')

# Create a dictionary to hold the data
data = {'train_images': train_images, 'train_labels': train_labels, 'test_images': test_images, 'test_labels': test_labels}

# Save the data to a pickle file
with open('mnist_dataset.pkl', 'wb') as f:
    pickle.dump(data, f)
