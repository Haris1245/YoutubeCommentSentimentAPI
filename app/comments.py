import os
import json
import googleapiclient.discovery
import googleapiclient.errors
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

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


def get_video_id(url):
    # Extract video ID from YouTube URL
    video_id = None
    if "youtube.com" in url or "youtu.be" in url:
        video_id = url.split("v=")[1] if "v=" in url else url.split("/")[-1]
    return video_id

def get_video_comments(api_key, video_id, max_results=100):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=max_results
    ).execute()
    
    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments

def main(url):
    # Replace 'YOUR_API_KEY' with your actual YouTube Data API key
    api_key = os.getenv('APIKEY')

    # Replace 'YOUR_VIDEO_URL' with the URL of the desired YouTube video
    video_id = url
    model = keras.saving.load_model("model.h5")
    class_names= ["Positive", "Negative"]
    results = []

    if video_id:
        comments = get_video_comments(api_key, video_id)

        for index, comment in enumerate(comments, start=1):
            line = " ".join(comment)
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
            predict = model.predict(encode)
            prediction = class_names[0] if predict[0][0] > 0.5 else class_names[1]

            result = {
                'index': index,
                'comment': comment,
                'prediction': prediction
            }
            results.append(result)


    else:
        print("Invalid YouTube video URL.")
    return results

if __name__ == "__main__":
    main()
