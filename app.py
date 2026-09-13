"""
Fire Segmentation Web App — Flask backend
Loads a trained YOLO26-seg model and runs it on an uploaded video,
then serves the annotated result back to a web frontend.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import uuid
import subprocess
import threading

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, url_for
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS    = os.path.join(BASE_DIR, "weights", "best.pt")   # <-- your trained weights
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIDEO_EXT  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED    = VIDEO_EXT | IMAGE_EXT
CONF       = 0.25          # detection confidence threshold
IMGSZ      = 640

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB max upload

# Load the model ONCE at startup (not per request) — this is the key thing.
print("Loading model:", WEIGHTS)
model = YOLO(WEIGHTS)
print("Model loaded. Classes:", model.names)

# Simple in-memory job tracker: job_id -> status dict
JOBS = {}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED


def media_kind(filename):
    """Return 'image' or 'video' based on the file extension."""
    return "image" if os.path.splitext(filename)[1].lower() in IMAGE_EXT else "video"


def tally(results):
    """Count segmented instances per class across all result frames."""
    total = 0
    per_class = {}
    for r in results:
        if r.masks is not None:
            for c in r.boxes.cls.tolist():
                name = model.names[int(c)]
                per_class[name] = per_class.get(name, 0) + 1
                total += 1
    return total, per_class


def process_media(job_id, in_path, out_name, kind):
    """Run YOLO segmentation on an image or a video and prepare the output."""
    try:
        JOBS[job_id]["status"] = "processing"

        # YOLO writes the annotated file into a per-job run folder.
        results = model.predict(
            source=in_path,
            conf=CONF,
            imgsz=IMGSZ,
            save=True,
            project=OUTPUT_DIR,
            name=job_id,
            exist_ok=True,
            stream=False,
            verbose=False,
        )

        run_dir = os.path.join(OUTPUT_DIR, job_id)
        final_path = os.path.join(OUTPUT_DIR, out_name)

        if kind == "image":
            # Find the annotated image and copy it to the served location.
            produced = None
            for f in os.listdir(run_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    produced = os.path.join(run_dir, f)
                    break
            if produced is None:
                raise RuntimeError("YOLO did not produce an output image.")
            import shutil
            shutil.copy(produced, final_path)

        else:  # video
            produced = None
            for f in os.listdir(run_dir):
                if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm", ".mov")):
                    produced = os.path.join(run_dir, f)
                    break
            if produced is None:
                raise RuntimeError("YOLO did not produce an output video.")

            # Re-encode to H.264 so browsers can play it inline. If ffmpeg
            # isn't installed, fall back to serving YOLO's raw output.
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", produced,
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_path],
                    check=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                import shutil
                shutil.copy(produced, final_path)
                print("WARNING: ffmpeg not available; serving raw YOLO output. "
                      "Install ffmpeg for reliable in-browser playback.")

        total, per_class = tally(results)

        JOBS[job_id].update(
            status="done",
            kind=kind,
            output=out_name,
            total_detections=total,
            per_class=per_class,
        )
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Accept the file under either 'media' (new) or 'video' (backward-compatible).
    file = request.files.get("media") or request.files.get("video")
    if file is None:
        return jsonify(error="No file part"), 400
    if file.filename == "":
        return jsonify(error="No file selected"), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Use an image or a video."), 400

    kind = media_kind(file.filename)
    job_id = uuid.uuid4().hex[:12]
    safe = secure_filename(file.filename)
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe}")
    file.save(in_path)

    ext = ".jpg" if kind == "image" else ".mp4"
    out_name = f"{job_id}_result{ext}"
    JOBS[job_id] = {"status": "queued", "kind": kind}

    # Process in a background thread so the request returns immediately.
    t = threading.Thread(target=process_media,
                         args=(job_id, in_path, out_name, kind))
    t.start()

    return jsonify(job_id=job_id, kind=kind)


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    resp = dict(job)
    if job.get("status") == "done":
        # 'result_url' works for both images and videos; keep 'video_url' too
        # so older frontends don't break.
        url = url_for("result_file", filename=job["output"])
        resp["result_url"] = url
        resp["video_url"] = url
    return jsonify(resp)


@app.route("/outputs/<path:filename>")
def result_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # threaded=True lets the status endpoint respond while a video is processing.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)