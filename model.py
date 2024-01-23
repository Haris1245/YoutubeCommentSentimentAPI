import tensorflow as tf
import numpy as np
from tensorflow import keras 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = keras.datasets.imdb 

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=98000)
word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
        return " ".join([reverse_word_index.get(i, "?")for i in text])

model = keras.Sequential()
model.add(keras.layers.Embedding(98000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]
fitModel = model.fit(x_train, y_train, epochs=150,batch_size=128, validation_data=(x_val, y_val), verbose=1)
loss,acc = model.evaluate(test_data, test_labels)
print(f"Accuracy: {round(acc, 4)}, Loss: {round(loss, 4)}" )
model.save("model.h5")
# def review_encode(s):
# 	encoded = [1]

# 	for word in s:
# 		if word.lower() in word_index:
# 			encoded.append(word_index[word.lower()])
# 		else:
# 			encoded.append(2)

# 	return encoded
        

# model = keras.saving.load_model("modeel.h5")
# class_names= ["Positive", "Negative"]
# with open("test.txt", encoding="utf-8") as f:
#     line = " ".join(f.readlines())
#     nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
#     encode = review_encode(nline)
#     encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
#     predict = model.predict(encode)
#     print(line)
#     if predict[0][0] > 0.5:
#         print(class_names[0])
#     elif predict[0][0] < 0.5:
#         print(class_names[1])

# class_names = ["Positive", "Negative"]
# test_review =test_data[0]
# prediction  = model.predict([test_review])
# print(f"Review: {decode_review(test_review)}")
# print(f"Prediction: {str(prediction[0])}")
# print(f"Actual: {str(test_labels[0])}")
