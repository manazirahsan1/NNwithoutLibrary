import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

model_params = {}
gradients = {}
interim_values = {}

def load_dataset():
    # test_data = pd.read_csv("https://pjreddie.com/media/files/mnist_test.csv")
    train_data = pd.read_csv("https://pjreddie.com/media/files/mnist_train.csv")
    train_data = np.array(train_data)
    m, n = train_data.shape
    np.random.shuffle(train_data)
    train_data = train_data[0:m].T
    X_train = train_data[1:n] / 255
    Y_train = train_data[0]
    # X_train = X_train / 255
    return X_train, Y_train

def save_model(model_params, filename):
    with open(filename, 'wb') as f:
    	pickle.dump(model_params, f)

def load_model(filename):
    try:
    	with open(filename, 'rb') as f:
    	    model_params = pickle.load(f)
    	    print("Loaded model from:", filename)
    except FileNotFoundError:
    	print("Model not found. Initializing new model!")
    	model_params = init_params()
    return model_params

# Model: [784, 256, 128, 32, 10]
def init_params():
    model_params["w1"] = np.random.rand(256, 784) - 0.5
    model_params["b1"] = np.zeros((256, 1)) - 0.5
    model_params["w2"] = np.random.rand(64, 256) - 0.5
    model_params["b2"] = np.zeros((64, 1)) - 0.5
    model_params["w3"] = np.random.rand(32, 64) - 0.5
    model_params["b3"] = np.zeros((32, 1)) - 0.5
    model_params["w4"] = np.random.rand(10, 32) - 0.5
    model_params["b4"] = np.zeros((10, 1)) - 0.5
    return model_params

def relu(z):
    return np.maximum(z,0) # np.maximum(z,0) is ReLU

def relu_deriv(z):
    return z > 0

def softmax(z):
    return np.exp(z)/ sum(np.exp(z))

def forward(model_params, X):
    interim_values["z1"] = model_params["w1"].dot(X) + model_params["b1"]
    interim_values["a1"] = relu(interim_values["z1"])
    interim_values["z2"] = model_params["w2"].dot(interim_values["a1"]) + model_params["b2"]
    interim_values["a2"] = relu(interim_values["z2"])
    interim_values["z3"] = model_params["w3"].dot(interim_values["a2"]) + model_params["b3"]
    interim_values["a3"] = relu(interim_values["z3"])
    interim_values["z4"] = model_params["w4"].dot(interim_values["a3"]) + model_params["b4"]
    interim_values["a4"] = softmax(interim_values["z4"])
    return interim_values

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def cat_cross_entropy(one_hot_Y, a4):
    n, m = one_hot_Y.shape
    CCE = -np.sum(one_hot_Y * np.log(a4)) * 1/m
    return CCE

def back_prop(model_params, interim_values, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    gradients["dz4"] = interim_values["a4"] - one_hot_Y
    gradients["dw4"] = gradients["dz4"].dot(interim_values["a3"].T) * 1/m
    gradients["db4"] = np.sum(gradients["dz4"]) * 1/m
    gradients["dz3"] = relu_deriv(interim_values["z3"]) * model_params["w4"].T.dot(gradients["dz4"])
    gradients["dw3"] = gradients["dz3"].dot(interim_values["a2"].T) * 1/m
    gradients["db3"] = np.sum(gradients["dz3"]) * 1/m
    gradients["dz2"] = relu_deriv(interim_values["z2"]) * model_params["w3"].T.dot(gradients["dz3"])
    gradients["dw2"] = gradients["dz2"].dot(interim_values["a1"].T) * 1/m
    gradients["db2"] = np.sum(gradients["dz2"]) * 1/m
    gradients["dz1"] = relu_deriv(interim_values["z1"]) * model_params["w2"].T.dot(gradients["dz2"])
    gradients["dw1"] = gradients["dz1"].dot(X.T) * 1/m
    gradients["db1"] = np.sum(gradients["dz1"]) * 1/m
    return gradients

def update_params(model_params, gradients, learning_rate):
    model_params["w1"] = model_params["w1"] - learning_rate * gradients["dw1"]
    model_params["b1"] = model_params["b1"] - learning_rate * gradients["db1"]
    model_params["w2"] = model_params["w2"] - learning_rate * gradients["dw2"]
    model_params["b2"] = model_params["b2"] - learning_rate * gradients["db2"]
    model_params["w3"] = model_params["w3"] - learning_rate * gradients["dw3"]
    model_params["b3"] = model_params["b3"] - learning_rate * gradients["db3"]
    model_params["w4"] = model_params["w4"] - learning_rate * gradients["dw4"]
    model_params["b4"] = model_params["b4"] - learning_rate * gradients["db4"]
    return model_params

def get_predictions(a2):
    predictions = np.argmax(a2, axis = 0)
    return predictions

def accuracy(predictions, Y):
    acc = np.sum(predictions == Y) / Y.size
    return acc

def gradient_descent(model_params, X, Y, learning_rate, epochs):
    for i in range(epochs):
    	interim_values = forward(model_params, X)
    	gradients = back_prop(model_params, interim_values, X, Y)
    	model_params = update_params(model_params, gradients, learning_rate)
    	if (i + 1) % 10 == 0:
    	    loss = cat_cross_entropy(one_hot(Y), interim_values["a4"])
    	    predictions = get_predictions(interim_values["a4"])
    	    acc = accuracy(predictions, Y) * 100
    	    print(f"Epoch: {i + 1}\tAccuracy: {acc:.2f}%\tLoss: {loss}")
    return model_params

filename = 'mnistnn.pkl'
learning_rate = 0.1
n_epochs = 100
X_train, Y_train = load_dataset()
model_params = load_model(filename)
model_params = gradient_descent(model_params, X_train, Y_train, learning_rate, n_epochs)
save_model(model_params, filename)

