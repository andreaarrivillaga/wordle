import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import pickle

# Load the word list
with open("words2.csv") as f:
    word_list = [word.strip().lower() for word in f.readlines() if len(word.strip()) == 5]

print(f"Total valid 5-letter words: {len(word_list)}")


# Define the custom vectorize function (must be defined here for pickle to work)
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


# Define the model
class WordleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(WordleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    model = WordleNN(input_size=130, output_size=15)  # Adjust input_size/output_size as needed
    try:
        model.load_state_dict(torch.load("wordle_nn.pth"))
        model.eval()
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved as 'wordle_nn.pth'.")
    return model


# Filter valid words based on feedback
def filter_valid_words(valid_words, guesses):
    """
    Filter the valid words based on the feedback from previous guesses.
    """
    for guess, feedback in guesses:
        for i, (color, letter) in enumerate(feedback):
            if color == "green":
                # Keep only words with this letter in the correct position
                valid_words = [word for word in valid_words if word[i] == letter]
            elif color == "yellow":
                # Keep only words that contain this letter but not in this position
                valid_words = [word for word in valid_words if letter in word and word[i] != letter]
            elif color == "gray":
                # Remove words that contain this letter unless it appears as green or yellow in another position
                other_feedback = [fb for j, fb in enumerate(feedback) if j != i]
                if all(fb[1] != letter for fb in other_feedback):
                    valid_words = [word for word in valid_words if letter not in word]
    return valid_words


def give_feedback(guess, target_word):
    """Provide feedback for the guess."""
    feedback = []  # (color, letter)
    target_word_list = list(target_word)

    for i, char in enumerate(guess):
        if char == target_word[i]:
            feedback.append(('green', char))
            target_word_list[i] = None  # Remove matched letters
        elif char in target_word_list:
            feedback.append(('yellow', char))
            target_word_list[target_word_list.index(char)] = None
        else:
            feedback.append(('gray', char))

    return feedback


def suggest_next_word(model, valid_words, guesses):
    """
    Suggest the next word using the model and filtered valid words.
    """
    # Filter valid words based on feedback
    valid_words = filter_valid_words(valid_words, guesses)

    if not valid_words:
        return "NO SUGGESTIONS", 0.0

    # Vectorize the valid words
    try:
        vectorized = custom_vectorize(valid_words)
        vectorized_tensor = torch.tensor(vectorized, dtype=torch.float32)
    except Exception as e:
        st.error(f"Vectorization failed: {e}")
        return "NO SUGGESTIONS", 0.0

    # Model inference
    try:
        outputs = model(vectorized_tensor)
        probabilities = torch.softmax(outputs.view(-1, 3), dim=1)[:, 2]
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        return "NO SUGGESTIONS", 0.0

    if probabilities.numel() == 0:
        return "NO SUGGESTIONS", 0.0

    # Get the best suggestion
    best_index = torch.argmax(probabilities).item() // 5
    return valid_words[best_index], probabilities[best_index * 5].item()


def reset_game():
    """Initialize or reset the game state."""
    if "game_initialized" not in st.session_state:
        st.session_state.attempts = 6
        st.session_state.guesses = []
        st.session_state.current_guess = ""
        st.session_state.chosen_word = random.choice(word_list)
        st.session_state.valid_words = word_list.copy()
        st.session_state.suggested_words = []
        st.session_state.game_initialized = True


# Streamlit UI
st.title("Wordle Game - Neural Network Suggestions")
st.write("Enter a 5-letter word and get feedback with suggestions!")

# Load the model
model = load_model()
reset_game()

# Function to clear the input box after submission
def submit():
    st.session_state.my_text = st.session_state.widget
    st.session_state.widget = ""

# Ensure game state is reset on reload
if 'game_initialized' not in st.session_state or not st.session_state.game_initialized:
    reset_game()

# Text input for guesses with clearing functionality
if "my_text" not in st.session_state:
    st.session_state.my_text = ""

st.text_input("Type your guess (5 letters):", max_chars=5, key="widget", on_change=submit)
current_guess=st.session_state.my_text

# Set guess input to what was in the text box before clearing
guess_input = st.session_state.my_text

if current_guess and len(current_guess) == 5:
    # Ensure the word is valid
    if current_guess not in word_list:
        st.warning("Invalid word. Please enter a valid 5-letter word.")
    else:
        feedback = give_feedback(current_guess, st.session_state.chosen_word)
        st.session_state.guesses.append((current_guess, feedback))
        st.session_state.attempts -= 1

        # Filter valid words based on feedback
        st.session_state.valid_words = [
            word for word in st.session_state.valid_words if word != current_guess
        ]

        suggested_word, probability = suggest_next_word(model, st.session_state.valid_words, st.session_state.guesses)

        if suggested_word == "NO SUGGESTIONS":
            st.write("No suggestions available.")
        else:
            st.write(f"Suggested next word: {suggested_word.upper()} (Probability: {probability:.4f})")

# Display the game board
rows, cols = 6, 5
for i in range(rows):
    cols_display = st.columns(cols)
    if i < len(st.session_state.guesses):
        guess, feedback = st.session_state.guesses[i]
        for j in range(cols):
            color, letter = feedback[j]
            background_color = {"green": "#00FF00", "yellow": "#FFFF00", "gray": "#D3D3D3"}.get(color, "#FFFFFF")
            cols_display[j].markdown(
                f"<div style='text-align:center; background-color:{background_color}; color:black; border:1px solid black;'>{letter.upper()}</div>",
                unsafe_allow_html=True,
            )
    else:
        for j in range(cols):
            cols_display[j].markdown(
                f"<div style='text-align:center; color:black; border:1px solid black;'>&nbsp;</div>",
                unsafe_allow_html=True,
            )


# Remaining attempts
st.write(f"Remaining Attempts: {st.session_state.attempts}")

# Check for win/loss condition
if len(st.session_state.guesses) > 0 and all(color == "green" for color, _ in st.session_state.guesses[-1][1]):
    st.success("Congratulations! You guessed the word!")
    st.session_state.game_initialized = False
elif st.session_state.attempts == 0:
    st.error(f"Game Over! The correct word was: {st.session_state.chosen_word.upper()}")
    st.session_state.game_initialized = False
