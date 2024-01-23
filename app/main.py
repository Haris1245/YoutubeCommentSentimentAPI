from flask import Flask, jsonify
from comments import main
app = Flask(__name__)


@app.route('/<id>', methods=['GET'])
def index(id):
    results = main(id)
    comments = []
    predictions = []
    positives = 0
    for i in range(len(results)):
        result = {"comment": results[i]['comment'], "prediction": results[i]['prediction']}
        comments.append(result)
        prediction = {"prediction": results[i]['prediction']}
        predictions.append(prediction)
        if predictions[i]["prediction"] == "Positive":
             positives += 1
    print("Summary", (positives / len(comments)) * 100, "%")
    return jsonify(comments)



if __name__ == "__main__":
    app.run(debug=True)