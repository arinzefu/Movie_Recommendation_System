import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
import os
from keras.layers import LSTM, Embedding
from keras_preprocessing.sequence import pad_sequences


# Load the movies.csv file
movies = pd.read_csv('movies.csv')

# Clean the movie titles by removing extra whitespace and punctuation marks
movies['title'] = movies['title'].str.strip()
movies['title'] = movies['title'].str.replace('[^\w\s]', '')

# Load the ratings.csv file
ratings = pd.read_csv('ratings.csv')

# Drop the timestamp column from the ratings dataframe
ratings = ratings.drop('timestamp', axis=1)

# Check for any missing or duplicate data in the ratings dataframe
print('Missing values in ratings:\n', ratings.isnull().sum())
print('Duplicate values in ratings:', ratings.duplicated().sum())

# Load the tags.csv file
tags = pd.read_csv('tags.csv')

# Drop the timestamp column from the tags dataframe
tags = tags.drop('timestamp', axis=1)

# Check for any missing or duplicate data in the tags dataframe
print('Missing values in tags:\n', tags.isnull().sum())
print('Duplicate values in tags:', tags.duplicated().sum())

# Merge the movies, ratings, and tags data into a single DataFrame
data = pd.merge(movies, ratings, on='movieId')
data = pd.merge(data, tags, on=['userId', 'movieId'])

# Check for any missing or duplicate data in the merged dataframe
print('Missing values in data:\n', data.isnull().sum())
print('Duplicate values in data:', data.duplicated().sum())





# Convert movie titles to numerical IDs
movie_id_to_name = dict(zip(data['movieId'].unique(), range(len(data['movieId'].unique()))))
data['movieId'] = data['movieId'].map(movie_id_to_name)

# Convert user IDs to numerical IDs
user_id_to_name = dict(zip(data['userId'].unique(), range(len(data['userId'].unique()))))
data['userId'] = data['userId'].map(user_id_to_name)

# Split the data into training, validation, and testing sets
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


# Create a vocabulary of all unique movie IDs
vocabulary = sorted(data['movieId'].unique())

# Define the vocabulary size and sequence length
vocabulary_size = len(vocabulary)
max_sequence_length = 200
def create_sequences(data, vocabulary, max_sequence_length):
    sequences = []
    for user_id, group in data.groupby('userId'):
        movie_ids = group['movieId'].values
        sequence = [vocabulary.index(movie_id) + 1 for movie_id in movie_ids if movie_id in vocabulary]
        if len(sequence) > max_sequence_length:
            sequence = sequence[:max_sequence_length]
        else:
            sequence = sequence + [0] * (max_sequence_length - len(sequence))
        sequences.append(sequence)
    return np.array(sequences)

# Create sequences of movie IDs for each user
train_sequences = create_sequences(train_data, vocabulary, max_sequence_length)
val_sequences = create_sequences(val_data, vocabulary, max_sequence_length)
test_sequences = create_sequences(test_data, vocabulary, max_sequence_length)

print(train_sequences)
print(train_sequences.shape)

print("Minimum value:", np.min(train_sequences))
print("Maximum value:", np.max(train_sequences))
print(val_data.size)
print(val_sequences)
print(val_sequences.shape)
print("Minimum value:", np.min(val_sequences))
print("Maximum value:", np.max(val_sequences))
print(test_data.size)
print(test_sequences)
print(test_sequences.shape)
print("Minimum value:", np.min(test_sequences))
print("Maximum value:", np.max(test_sequences))




#Pad sequences with zeros to make them the same length
max_sequence_length = 200

train_sequences_padded = pad_sequences(train_sequences.tolist(), maxlen=max_sequence_length, padding='post', truncating='post')
val_sequences_padded = pad_sequences(val_sequences.tolist(), maxlen=max_sequence_length, padding='post', truncating='post')
test_sequences_padded = pad_sequences(test_sequences.tolist(), maxlen=max_sequence_length, padding='post', truncating='post')

test_max_movie_id = np.max(test_sequences_padded)
print(test_max_movie_id)

print(train_sequences_padded.shape)
print(val_sequences_padded.shape)

print(test_sequences_padded.shape)



# Define the RNN model


# Get the maximum movie ID

max_movie_id = data['movieId'].max()

# Define the embedding size
embedding_size = 64

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>") #to remove the out of vocabulary(OOV)

# Define integer encoding function
def integer_encode_sequences(sequences, num_movies):
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequence = []
        for movie_id in sequence:
            encoded_sequence.append(movie_id-1) # subtract 1 to make movie IDs start from 0
        encoded_sequences.append(encoded_sequence)
    return encoded_sequences

# Integer encode sequences
y_train = np.array(integer_encode_sequences([seq[:200] for seq in train_sequences_padded], max_movie_id))
y_val = np.array(integer_encode_sequences([seq[:200] for seq in val_sequences_padded], max_movie_id))
y_test = np.array(integer_encode_sequences([seq[:200] for seq in test_sequences_padded], max_movie_id))

X_train = np.array(integer_encode_sequences([seq[:200] for seq in train_sequences_padded], max_movie_id))
X_val = np.array(integer_encode_sequences([seq[:200] for seq in val_sequences_padded], max_movie_id))
X_test = np.array(integer_encode_sequences([seq[:200] for seq in test_sequences_padded], max_movie_id))
# 200 is the maximum number of sequences or max_sequence_length = 200


test_max_movie_id = np.max(test_sequences_padded)
print(test_max_movie_id)

y_train[y_train == -1] = max_movie_id+2
y_test[y_test == -1] = max_movie_id+2
y_val[y_val == -1] = max_movie_id+2

X_train[X_train == -1] = max_movie_id+2
X_test[X_test == -1] = max_movie_id+2
X_val[X_val == -1] = max_movie_id+2
# using max_movie_id +2 to replace the -1

lstm_size = 128

# Define the model
model = Sequential()
model.add(Embedding(max_movie_id+1, embedding_size, input_length=max_sequence_length, mask_zero=True))
model.add(LSTM(lstm_size, return_sequences=True))
model.add(Dense(max_movie_id+1, activation='softmax'))

#Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val), callbacks=[es_callback])

# Graph of the history

fig = plt.figure()
plt.plot(history.history['loss'], color='blue', label='loss')
plt.plot(history.history['val_loss'], color='green', label='val_loss')
fig.suptitle('LOSS OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='blue', label='accuracy')
plt.plot(history.history['val_accuracy'], color='green', label='val_accuracy')
fig.suptitle('ACCURACY OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()

# Evaluate the model on the test set

test_loss, test_acc = model.evaluate(y_test, X_test)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')


# Save the model
model.save(os.path.join('Model','MovieRecommendation_model.h5'))