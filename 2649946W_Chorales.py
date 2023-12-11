#!/usr/bin/env python
# coding: utf-8

# 2649946W  https://github.com/2649946w/2649946w.git

# # Generating Chorales With RNN

# # Getting the Data

# In[ ]:


#Importing the necessary database
import tensorflow as tf

tf.keras.utils.get_file(
    "jsb_chorales.tgz",
    "https://github.com/ageron/data/raw/main/jsb_chorales.tgz",
    cache_dir=".",
    extract=True)
#the above function downloads a file named "jsb_chorales.tgz" from a GitHub repository and then saves it to the local disk. 


# In[ ]:


from pathlib import Path
#This imports the Path class from the pathlib module in Python. This allows you to work with paths in a far more elegant and efficient way.

jsb_chorales_dir = Path("datasets/jsb_chorales")
train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))
#The purpose of this block of code is to organize the files in the JS Bach chorales dataset into training, validation, and test sets.


# In[ ]:


import pandas as pd
#importing the pandas dataset

def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]

train_chorales = load_chorales(train_files)
valid_chorales = load_chorales(valid_files)
test_chorales = load_chorales(test_files)
#this block determines which files are training data and which are test data


# # Preparing the Data

# In[ ]:


notes = set()
for chorales in (train_chorales, valid_chorales, test_chorales):
    for chorale in chorales:
        for chord in chorale:
            notes |= set(chord)
            #this block defines the notes to the programme, displaying 

n_notes = len(notes)
min_note = min(notes - {0}) #0 denotes no notes being played
max_note = max(notes)

assert min_note == 36
assert max_note == 81


# ### Code for Synthesiser
# 
# The following cell is code for a synthesiser to play MIDI. Not part of machine learning code to generate Bach, but useful for listening to the results and samples used for training!

# In[ ]:


from IPython.display import Audio
import numpy as np
#this allows for audio files to be displayed

def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440

def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = (note_duration * frequencies).round() / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.).reshape(-1, 1)
    return sine_waves.reshape(-1)

def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]] # make last note a bit longer
    merged = np.mean([frequencies_to_samples(melody, tempo, sample_rate)
                     for melody in freqs.T], axis=0)
    n_fade_out_samples = sample_rate * 60 // tempo # fade out last note
    fade_out = np.linspace(1., 0., n_fade_out_samples)**2
    merged[-n_fade_out_samples:] *= fade_out
    return merged

def play_chords(chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))

## testing the synthesiser
for index in range(3):
    play_chords(train_chorales[index])


# In[ ]:


from IPython.display import Image
Image ("piano.jpg")
#this is what the above code does:


# In[ ]:


import tensorflow as tf

def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:] # predict next note in each arpegio, at each step
    return X, Y

def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1) # shift values
    return tf.reshape(window, [-1]) # convert to arpegio

def bach_dataset(chorales, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)

# this block of code defines the dataset class BachDataset. This takes a collection of J.S. Bach's chorales as input and processes them into a 
#format suitable for training a neural network. The dataset class includes several parameters that control the preprocessing steps and the size 
#of the batches returned by the dataset.


# In[ ]:


train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
valid_set = bach_dataset(valid_chorales)
test_set = bach_dataset(test_chorales)
#This sets the sample size for the training data and defines that the training will be a random shuffle. It therefore has a bigger size than the test data
#valid_set and test_set have a fixed size based on the number of chorales in the corresponding lists.


# # Building the Model

# In[ ]:


n_embedding_dims = 5

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_notes, output_dim=n_embedding_dims,
                           input_shape=[None]),
    tf.keras.layers.Conv1D(32, kernel_size=2, padding="causal", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(48, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(n_notes, activation="softmax")
])
#this is the beautiful bulk of the code. defining and building the model  using the Keras API of TensorFlow.
# Specifically, it defines a sequential model consisting of several layers
#It intercuts many of the layers with batch normalisation layers that help stabalise and improve the nuerel network by
#transforming the input data into a normal distribution with a mean of 0 and a standard deviation of 1


# In[ ]:


model.summary()


# # Training the Model

# In[ ]:


optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=20, validation_data=valid_set)
#this is the code that begins the models training process which was defined earlier, think rocky running up the stairs in Rocky 1.


# In[ ]:


Image ("lebronmeme.jpg")
#this is how i felt upon seeing how long this would take to load


# In[ ]:


Image ("rockymeme.jpg")
#meanwhile this is what the code was doing


# # Saving and Evaluating Your Model 

# In[ ]:


model.save("my_bach_model", save_format="tf")
model.evaluate(test_set)
#these two lines operate different functions but in summary the code saves the trained model to a file named "my_bach_model" in the 
#current working directory and then evaluates the performance of the model on a test set, printing out the accuracy of the model on the test set.


# # Generating Chorales

# In[ ]:


def generate_chorale_v2(model, seed_chords, length, temperature=1):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio)[0, -1:]
            rescaled_logits = tf.math.log(next_note_probas) / temperature
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])
#This function generates a chorale by iteratively adding notes to a seed chord progression, as defined earlier in the code. 
#The model predicts the probabilities of the next note given the previous notes, and a random note is selected from these probabilities. 
#The resulting chorale is a four-part harmony with a specified length.


# In[ ]:


seed_chords = test_chorales[2][:8]
play_chords(seed_chords, amplitude=0.2)
#this plays the chorale with the volume at 0.2, it begins at the the root note of C and works up the scale from there 
#with a short pause inbetween each note


# In[ ]:


new_chorale_v2_cold = generate_chorale_v2(model, seed_chords, 56, temperature=0.8)
play_chords(new_chorale_v2_cold, filepath="bach_cold.wav")


# In[ ]:


new_chorale_v2_medium = generate_chorale_v2(model, seed_chords, 56, temperature=1.0)
play_chords(new_chorale_v2_medium, filepath="bach_medium.wav")


# In[ ]:


new_chorale_v2_hot = generate_chorale_v2(model, seed_chords, 56, temperature=1.5)
play_chords(new_chorale_v2_hot, filepath="bach_hot.wav")


# This code defines a machine learning program that is designed to generate chorales in the style of Johann Sebastian Bach. The program begins by importing the necessary libraries, as all good programmes should. Importantly TensorFlow is imported, which is used to train the machine learning model.
# Next, the code collects a dataset of existing chorales composed by Bach. Following this the data is sorted into training and test data for later on in the programme.
# 
# Then the sounds of the synthesizer is built, this allows for the machine to translate its predicted note into sound and display the output in a certain way.
# .
# 
# The machine learning model itselfbuiltined using TensorFlowis The mois made up of a  of mtude ofiple la.atAfter building the modelefined, it is trained usi aformentionedncollectionataset of Bach's choraliss. The training process invgiving the machineeedi raw g the inpu mand subsequently odel, adjusting the weights and biases of the model's paramet optomize the model anders to minimize erthis process is then repeated until the machine can accurately produce chorales
# ales.
# 
# After training, the model is evaluated to ensure th in factat it is accurately generating chorales in the style oThis fine-tunes the model allowing for the production of accurate Chorales ccuracy.
# 
# Fis e traigenerateso gene three new chorales in the style of Bach. The input data for these chorales is created by randomly selecting a key, tempo, and time signature,using what was learnt in the training stage to produce a piano roll and apply the synthezier to this sound.ouThe result looks like this:yle of Bach.

# In[ ]:


Image ("finalmeme.jpg")


# Hugging Chat was used to assist my understanding of the code. I also used the code seen earlier in the course to apply learnt knowledge in my understanding of this project. All of the fantastic memes came from my brain (and google)

# In[ ]:




