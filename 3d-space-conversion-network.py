#!/bin/python3

import json
import random
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the neural network architecture
input_nodes = 3
hidden_nodes = 10
output_nodes = 6

# Create the neural network model
model = models.Sequential([
    layers.Dense(hidden_nodes, activation='relu', input_shape=(input_nodes,)),
    layers.Dense(output_nodes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load training data from a JSON file
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

# Extract input and output data from training data
inputs = [data['input'] for data in training_data]
outputs = [data['output'] for data in training_data]

# Convert input and output data to numpy arrays
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

# Train the model
model.fit(inputs, outputs, epochs=3)

# Test the model with random input data
num_tests = 5  # Set the number of test data sets
for _ in range(num_tests):
    random_input = tf.random.uniform((1, input_nodes))  # Generate random input
    prediction = model.predict(random_input)  # Make a prediction
    print("Random Input:", random_input.numpy())
    print("Prediction:", prediction[0])
