import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# sklearn is used only to normalize the train and test data
from sklearn.preprocessing import MinMaxScaler


#  -------------- Read Data --------
# install the packages and openpyxl

data = pd.read_excel("Dry_Bean_Dataset.xlsx")

# fix missing values
bom_missing = data[data['Class'] == 'BOMBAY']
bom_missing['MinorAxisLength'] = bom_missing['MinorAxisLength'].fillna(bom_missing['MinorAxisLength'].mean())
data['MinorAxisLength'][3] = bom_missing['MinorAxisLength'][3]

# ---------- check box values ----------------------------
checkBoxValues = [*data]
checkBoxValues.pop(-1)
scaler = MinMaxScaler()


def signum(x):
    if x >= 0:
        return 1
    else:
        return -1


# Perceptron training function
def perceptron_train(x_train, y_train, learning_rate: float, num_epochs: int, add_bias):
    weights = np.ones(x_train.shape[1])  # Initialize weights randomly
    bias = 0
    for epoch in range(num_epochs):
        for x, y in zip(x_train, y_train):
            y_pred = np.dot(weights, x) + bias
            y_pred = signum(y_pred)
            error = y - y_pred
            if not y == y_pred:
                weights = weights + learning_rate * x * error
                if add_bias:
                    bias = bias + learning_rate * error

    if not add_bias:
        bias = 0
    return weights, bias


# Adaline training function
def adaline_train(x, y, learning_rate, num_epochs, mse_threshold, add_bias):
    num_features = x.shape[1]
    bias = 0
    weights = np.random.rand(num_features)
    for epoch in range(num_epochs):
        errors = []
        for i in range(len(x)):
            xi = x[i]
            ti = y[i]
            yi = np.dot(weights, xi) + bias
            error = ti - yi
            weights = weights + learning_rate * error * xi
            if add_bias:
                bias = bias + learning_rate * error
            errors.append(abs(error ** 2) / 2)
        mse = np.mean(errors)
        if mse < mse_threshold:
            print(f"Converged after {epoch + 1} epochs with MSE: {mse}")
            break
    if not add_bias:
        bias = 0
    return weights, bias


# ------------------------------------

# sliced data function that returns normalized test and train data
def sliced_data(x, y, s1, s2):
    selected_classes = [x, y]
    class1_train = data[data['Class'] == selected_classes[0]].sample(n=30, random_state=42)
    class2_train = data[data['Class'] == selected_classes[1]].sample(n=30, random_state=42)
    train = pd.concat([class1_train, class2_train])

    class1_test = data[data['Class'] == selected_classes[0]].drop(class1_train.index).sample(n=20, random_state=42)
    class2_test = data[data['Class'] == selected_classes[1]].drop(class2_train.index).sample(n=20, random_state=42)
    test = pd.concat([class1_test, class2_test])

    train = train.sample(frac=1, random_state=42)
    test = test.sample(frac=1, random_state=42)

    train = train.to_numpy()
    test = test.to_numpy()

    x_train = train[:, [checkBoxValues.index(s1), checkBoxValues.index(s2)]]
    y_train = train[:, -1]
    for i in range(60):
        if y_train[i] == selected_classes[0]:
            y_train[i] = 1
        else:
            y_train[i] = -1
    x_test = test[:, [checkBoxValues.index(s1), checkBoxValues.index(s2)]]
    y_test = test[:, -1]
    for i in range(40):
        if y_test[i] == selected_classes[0]:
            y_test[i] = 1
        else:
            y_test[i] = -1
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return x_train, y_train, x_test, y_test


# main
root = tk.Tk()
root.title("Dry Beans Classification")


# plotting and testing
def plot(weights, x_test, test_labels, bias, s1, s2):
    plt.figure(figsize=(8, 6))

    # Plot the decision boundary
    w1, w2 = weights[0], weights[1]
    x_min, x_max = x_test[:, 0].min(), x_test[:, 0].max()
    line_x = np.array([x_min, x_max])
    line_y = (-w1 * line_x - bias) / w2
    plt.plot(line_x, line_y, color='black', label='Decision Boundary', linestyle='--')

# Calculate the confusion matrix and accuracy --------------------------------
    tp = tn = fp = fn = 0
    for i, x in enumerate(x_test):
        prediction = np.dot(weights, x) + bias
        prediction = signum(prediction)

        if prediction == 1:
            if int(test_labels[i]) == 1:
                tp += 1
            else:
                fp += 1
        else:
            if int(test_labels[i]) == -1:
                tn += 1
            else:
                fn += 1
    plt.scatter(x_test[:, 0], x_test[:, 1], c=test_labels)

    # Display the confusion matrix
    confusion_matrix = [[tn, fp], [fn, tp]]
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Calculate the accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    plt.xlabel(s1)
    plt.ylabel(s2)
    plt.title(f"Accuracy: {accuracy * 100:.2f}%")
    plt.show()


# Function to start the classification
def classify_data():
    # Retrieve Input
    selected_features1 = feature_var1.get()
    selected_features2 = feature_var2.get()
    selected_classes = class_var.get()
    learning_rate = float(learning_rate_entry.get())
    num_epochs = int(epochs_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_var.get()
    selected_algorithm = algorithm_var.get()

    # get the 2 specific classes
    x = selected_classes.split(' & ')[0]
    y = selected_classes.split(' & ')[1]
    x_train, y_train, x_test, y_test = sliced_data(x, y, selected_features1, selected_features2)

    if selected_algorithm == "perceptron":
        weights, bias = perceptron_train(x_train, y_train, learning_rate, num_epochs, add_bias)
        plot(weights, x_test, y_test, bias, selected_features1, selected_features2)

    elif selected_algorithm == "adaline":
        weights, bias = adaline_train(x_train, y_train, learning_rate, num_epochs, mse_threshold, add_bias)
        plot(weights, x_test, y_test, bias, selected_features1, selected_features2)


# Feature selection
feature_label = ttk.Label(root, text="Select Two Features:")
feature_label.pack()
feature_var1 = tk.StringVar()
feature_var2 = tk.StringVar()
feature_var1.set('Area')
feature_var2.set('roundnes')
feature_combobox = ttk.Combobox(root, textvariable=feature_var1, values=checkBoxValues, state="readonly")
feature_combobox.pack()
feature_combobox2 = ttk.Combobox(root, textvariable=feature_var2, values=checkBoxValues, state="readonly")
feature_combobox2.pack()

# Class selection
class_label = ttk.Label(root, text="Select Two Classes:")
class_label.pack()
class_var = tk.StringVar()
class_var.set("BOMBAY & CALI")
class_combobox = ttk.Combobox(root, textvariable=class_var, values=["BOMBAY & CALI", "BOMBAY & SIRA", "CALI & SIRA"],
                              state="readonly")
class_combobox.pack()

# Learning rate input
learning_rate_label = ttk.Label(root, text="Enter Learning Rate (eta):")
learning_rate_label.pack()
learning_rate_var = tk.StringVar()
learning_rate_var.set(0.01)
learning_rate_entry = ttk.Entry(root, textvariable=learning_rate_var)
learning_rate_entry.pack()

# Number of epochs input
epochs_label = ttk.Label(root, text="Enter Number of Epochs (m):")
epochs_label.pack()
epochs_var = tk.StringVar()
epochs_var.set(100)
epochs_entry = ttk.Entry(root, textvariable=epochs_var)
epochs_entry.pack()

# MSE threshold input
mse_threshold_label = ttk.Label(root, text="Enter MSE Threshold:")
mse_threshold_label.pack()
mse_threshold_var = tk.StringVar()
mse_threshold_var.set(0.03)
mse_threshold_entry = ttk.Entry(root, textvariable=mse_threshold_var)
mse_threshold_entry.pack()

# Bias input
bias_var = tk.BooleanVar()
bias_var.set(True)
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.pack()

# Algorithm selection
algorithm_label = ttk.Label(root, text="Choose Algorithm:")
algorithm_label.pack()
algorithm_var = tk.StringVar()
algorithm_var.set("perceptron")
algorithm_perceptron = ttk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="perceptron")
algorithm_adaline = ttk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="adaline")
algorithm_perceptron.pack()
algorithm_adaline.pack()

# button
classify_button = ttk.Button(root, text="Classify Data", command=classify_data)
classify_button.pack()


root.mainloop()
