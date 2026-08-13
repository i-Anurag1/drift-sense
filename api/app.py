import io
import os
import sys

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_sense.matcher import locate_reference

app = Flask(__name__)
MAX_UPLOAD_MB = 15
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def read_gray(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "drift-sense"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/locate", methods=["POST"])
def locate():
    if "reference" not in request.files or "search" not in request.files:
        return jsonify({"error": "Both 'reference' and 'search' files are required"}), 400

    try:
        ref = read_gray(request.files["reference"])
        search = read_gray(request.files["search"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = locate_reference(ref, search)

    response = {
        "x": round(result.x, 2),
        "y": round(result.y, 2),
        "score": round(result.score, 4),
        "ambiguous_site_detected": result.ambiguous,
        "candidate_count": result.n_candidates,
    }

    if request.args.get("visualize") == "1":
        overlay = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(overlay, (int(result.x), int(result.y)), (0, 255, 0),
                        cv2.MARKER_CROSS, 28, 2)
        cv2.circle(overlay, (int(result.x), int(result.y)), 16, (0, 255, 0), 2)
        _, buf = cv2.imencode(".png", overlay)
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
