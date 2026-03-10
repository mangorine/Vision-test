from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

DATA_PATH = os.path.join("data")
actions = np.array(["67", "idle"])
no_sequences = 20
sequence_length = 60

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

log_dir = os.path.join("logs")
tensorboard_callback = TensorBoard(log_dir=log_dir)

checkpoint_callback = ModelCheckpoint(
    filepath='model_67.keras',
    monitor='loss',
    save_best_only=True,
    verbose=1
)

# Model LSTM
model = Sequential()

model.add(LSTM(32, return_sequences=True, activation='tanh', input_shape=(sequence_length, 132)))

model.add(LSTM(64, return_sequences=True, activation='tanh'))

model.add(LSTM(32, return_sequences=False, activation='tanh'))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam',
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy']
              )

model.fit(X_train, y_train,
          epochs=100,
          callbacks=[tensorboard_callback, checkpoint_callback])

print(model.summary())
print("FIN DE L'ENTRAÎNEMENT")
