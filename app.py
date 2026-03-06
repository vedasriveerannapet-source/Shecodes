from flask import Flask, request, jsonify
import os
from video_utils import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/analyze", methods=["POST"])
def analyze_video():

    video = request.files["video"]

    reference_text = request.form["reference_text"]

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    video.save(video_path)

    result = process_video(video_path, reference_text)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)