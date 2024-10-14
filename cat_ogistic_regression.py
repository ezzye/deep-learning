import numpy as np
import h5py
import matplotlib.pyplot as plt

# 1. Load the dataset from H5 files
def load_dataset():
    """
    Loads the cat/non-cat dataset from H5 files.

    Returns:
    train_set_x_orig -- Training set features (images)
    train_set_y -- Training set labels
    test_set_x_orig -- Test set features (images)
    test_set_y -- Test set labels
    classes -- Array of class labels
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # Train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # Train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])     # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])     # Test set labels

    classes = np.array(test_dataset["list_classes"][:])           # List of classes

    train_set_y_orig = train_set_y_orig.reshape((1, -1))          # Reshape labels
    test_set_y_orig = test_set_y_orig.reshape((1, -1))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Load the dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 2. Preprocess the data

# Get the number of training and test examples
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]

# Reshape the training and test examples
# Each image is reshaped to a vector of size (num_px * num_px * 3, 1)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T  # Shape: (num_px*num_px*3, m_train)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T     # Shape: (num_px*num_px*3, m_test)

# Standardize the data to have feature values between 0 and 1
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# 3. Helper functions

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or NumPy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
    Initialize w and b to zeros.

    Arguments:
    dim -- Size of the w vector (number of parameters)

    Returns:
    w -- Initialized vector of shape (dim, 1)
    b -- Initialized scalar (bias)
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    """
    Implement forward and backward propagation.

    Arguments:
    w -- Weights, a NumPy array of size (number of features, 1)
    b -- Bias, a scalar
    X -- Data of size (number of features, number of examples)
    Y -- True "label" vector (1 if cat, 0 if non-cat)

    Returns:
    grads -- Dictionary containing gradients of the weights and bias
    cost -- Cost function (logistic loss)
    """
    m = X.shape[1]  # Number of examples

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)            # Compute activation
    cost = -1/m * np.sum(Y * np.log(A) +       # Compute cost
                         (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = 1/m * np.dot(X, (A - Y).T)            # Gradient of weights
    db = 1/m * np.sum(A - Y)                   # Gradient of bias

    cost = np.squeeze(cost)  # Ensure cost is a scalar

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Optimize w and b using gradient descent.

    Arguments:
    w -- Weights
    b -- Bias
    X -- Input data
    Y -- Labels
    num_iterations -- Number of iterations for optimization
    learning_rate -- Learning rate of gradient descent
    print_cost -- If True, print cost every 100 iterations

    Returns:
    params -- Dictionary containing the weights and bias
    grads -- Dictionary containing gradients
    costs -- List of costs during optimization
    """
    costs = []

    for i in range(num_iterations):
        # Calculate gradients and cost
        grads, cost = propagate(w, b, X, Y)

        # Retrieve gradients
        dw = grads["dw"]
        db = grads["db"]

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record and print cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w,
              "b": b}

    return params, grads, costs

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned parameters.

    Arguments:
    w -- Weights
    b -- Bias
    X -- Data to predict

    Returns:
    Y_prediction -- Predictions for X
    """
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5).astype(int)  # Convert probabilities to 0 or 1
    return Y_prediction

# 4. Build the model

def model(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Build and train the logistic regression model.

    Arguments:
    X_train -- Training data
    Y_train -- Training labels
    X_test -- Test data
    Y_test -- Test labels
    num_iterations -- Number of iterations
    learning_rate -- Learning rate
    print_cost -- If True, print cost every 100 iterations

    Returns:
    d -- Dictionary containing information about the model
    """
    # Initialize parameters
    n_features = X_train.shape[0]
    w, b = initialize_with_zeros(n_features)

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train,
                                        num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    # Predict on training and test sets
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("Train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

# 5. Train the model

# Set hyperparameters
num_iterations = 2000
learning_rate = 0.005

# Train the logistic regression model
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y,
                                  num_iterations, learning_rate, print_cost=True)

# 6. Plot the cost function

costs = logistic_regression_model['costs']
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title(f"Learning rate = {learning_rate}")
plt.show()

# 7. Display test images with predictions

def display_test_predictions(indexes):
    """
    Displays test images along with their true and predicted labels.

    Arguments:
    indexes -- List of indexes of test images to display
    """
    for index in indexes:
        plt.figure()
        plt.imshow(test_set_x_orig[index])
        plt.title(f"True label: {classes[test_set_y[0, index]].decode('utf-8')}, " +
                  f"Predicted: {classes[int(logistic_regression_model['Y_prediction_test'][0, index])].decode('utf-8')}")
        plt.show()

# Example: Display predictions for first 5 test images
display_test_predictions([0, 1, 2, 3, 4])

# 8. Test with your own image (Optional)

def predict_my_image(image_path):
    """
    Predicts whether an image contains a cat or not.

    Arguments:
    image_path -- Path to the image file
    """
    from PIL import Image
    from scipy import ndimage

    # Preprocess the image to fit the model
    num_px = train_set_x_orig.shape[1]  # Image size (height and width)
    image = np.array(Image.open(image_path).resize((num_px, num_px)))
    plt.imshow(image)
    plt.show()

    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
          classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")

# Uncomment the following line to test with your own image
# predict_my_image("images/my_image.jpg")
