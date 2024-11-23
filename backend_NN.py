import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

# Load word list
with open("words2.csv") as f:
    word_list = [word.strip().lower() for word in f.readlines() if len(word.strip()) == 5]

#this is just to know how many 4 letter words are in the csv file
print(f"Total valid 5-letter words: {len(word_list)}")

# Generate training data
def generate_training_data(word_list):
    #data = list of words that serve as the model's input 
    #labels = A list of feedback corresponding to the guesses, represented numerically as 2 (green), 1 (yellow), and 0 (gray).
    data, labels = [], []
    for target_word in random.sample(word_list, 500):  # Randomly sample target words
        for word in word_list:
            feedback = []
            target_list = list(target_word)
            for i in range(5):
                if word[i] == target_word[i]:
                    feedback.append(2)  # Green
                    target_list[i] = None
                elif word[i] in target_list:
                    feedback.append(1)  # Yellow
                    target_list[target_list.index(word[i])] = None
                else:
                    feedback.append(0)  # Gray
            data.append(word)
            labels.append(feedback)
    return data, labels

# Vectorize words
def custom_vectorize(word_list):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    vectorized = []
    for word in word_list:
        word_vector = []
        for letter in word:
            letter_vector = [0] * 26
            letter_vector[alphabet.index(letter)] = 1
            word_vector.extend(letter_vector)  # Append the one-hot vector for each letter
        vectorized.append(word_vector)  # Resulting vector will be of size 130
    return np.array(vectorized)

# Generate training data
training_words, training_labels = generate_training_data(word_list)
X = custom_vectorize(training_words)  # Shape: (n_samples, 130) (130 because each word is made of 5 letter * 26 (letters))
y = torch.tensor(training_labels, dtype=torch.long)  # Shape: (n_samples, 5) 


# Save the custom vectorize function for frontend use
# https://docs.python.org/3/library/pickle.html
with open("wordle_vectorizer.pkl", "wb") as f:
    pickle.dump(custom_vectorize, f)
print("Vectorizer saved.")


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# Define the model
class WordleNN(nn.Module):
    def __init__(self):
        super(WordleNN, self).__init__()
        self.fc1 = nn.Linear(26 * 5, 128)  # Input size: 130, linear layer 1
        self.fc2 = nn.Linear(128, 15)  # Output: 5 positions * 3 states, linear layer 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).view(-1, 5, 3)  # Reshape output for 5 positions with 3 states each

# Initialize model, loss, and optimizer
model = WordleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training the model...")
batch_size = 64
num_epochs = 10 #Epochs = how many times this thing has to run

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, len(X_tensor), batch_size):
        # Get batch
        batch_X = X_tensor[i:i + batch_size]
        batch_y = y[i:i + batch_size]

        # Flatten labels for loss calculation
        batch_y_flat = batch_y.view(-1)

        # Forward pass
        optimizer.zero_grad()# you have to zero up the gradient
        outputs = model(batch_X)  # Output shape: (batch_size, 5, 3)
        outputs_flat = outputs.view(-1, 3)  # Flatten predictions for loss function

        # Compute loss
        loss = criterion(outputs_flat, batch_y_flat) 
        loss.backward() # calculates all of my deltas
        optimizer.step() #takes all of does deltas  + learning rates

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(X_tensor)
    #This just helps me monitor the model's training progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the model
#https://pytorch.org/docs/stable/generated/torch.save.html
#Saves the model's parameters (weights and biases)
torch.save(model.state_dict(), "wordle_nn.pth")

#see trained model.state_dict() variables in human readable form. 
# Print optimizer's state_dict
# print("Model's state_dict:")
# for var_name in model.state_dict():
#     print(var_name, "\t", model.state_dict()[var_name])

print("Model saved.")
