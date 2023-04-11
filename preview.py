import pickle
import matplotlib.pyplot as plt

# load the dataset from the pickle file
with open('mnist_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# extract the images and labels
train_images = dataset['train_images']
train_labels = dataset['train_labels']

# view a sample image
plt.imshow(train_images[2], cmap='gray')
plt.show()
