from flask import Flask, request, jsonify
import pymongo
import gridfs
from bson import ObjectId
import traceback
from predicted import predict_emotion  # This is your model function

app = Flask(__name__)

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["audioDB"]
fs = gridfs.GridFS(db, collection="audios")

@app.route('/predict', methods=['POST'])
def predict_emotion_route():  # 👈 renamed this function
    data = request.get_json()
    file_id = data.get('fileId')

    print(f"\nReceived file ID: {file_id}")

    if not file_id:
        return jsonify({"error": "No file ID provided"}), 400

    try:
        file_id = ObjectId(file_id)
        file = fs.get(file_id)
        print("Audio file found in GridFS")

        audio_data = file.read()
        print("Audio data read successfully")

        # Now this correctly calls the model function
        emotion = predict_emotion(audio_data)
        print(f"Predicted emotion: {emotion}")

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
