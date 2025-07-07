import base64
import os
import pickle
import tempfile
from io import BytesIO

import matplotlib
import redis
from flask import Flask, jsonify, request

matplotlib.use("Agg")
import sys

# Ensure the modules directory is in the Python path for backend deployment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
from flask_cors import CORS

from modules.create_bar_animation import create_bar_animation, dpi
from modules.create_bar_plot import plot_final_frame
from modules.data_processing import (
    extract_json_from_zip,
    fetch_and_process_files,
    prepare_df_for_visual_anims,
    prepare_df_for_visual_plots,
)
from modules.normalize_inputs import normalize_inputs
from modules.prepare_visuals import error_logged, image_cache

app = Flask(__name__)
CORS(app)

redis_url = os.environ.get("REDIS_URL")
r = redis.from_url(redis_url, decode_responses=False)


def store_session_data(session_id, df):
    """Store DataFrame in Redis with 1-hour expiration."""
    try:
        pickled_df = pickle.dumps(df)
        r.setex(session_id, 3600, pickled_df)
        return True
    except Exception as e:
        print(f"Redis store error: {e}")
        return False


def get_session_data(session_id):
    """Retrieve DataFrame from Redis."""
    try:
        pickled_df = r.get(session_id)
        if pickled_df:
            return pickle.loads(pickled_df)
        return None
    except Exception as e:
        print(f"Redis get error: {e}")
        return None


@app.route("/process", methods=["POST"])
def process_zip():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            uploaded_file.save(zip_path)

            json_contents = extract_json_from_zip(zip_path)

            if not json_contents:
                return jsonify({"error": "No streaming history JSON files found"}), 400

            df = fetch_and_process_files(json_contents)

            if df.empty:
                return jsonify({"error": "No data processed from files"}), 400
            # Store the DataFrame for later use
            session_id = f"session_{hash(str(df.iloc[0].to_dict()))}"
            if store_session_data(session_id, df):
                response_data = {
                    "data": df.to_dict(orient="records"),
                    "session_id": session_id,
                }
                return jsonify(response_data), 200
            else:
                return jsonify({"error": "Failed to store session data"}), 500

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        # request data
        data = request.get_json()
        session_id = data.get("session_id")
        selected_attribute = data.get("selected_attribute")
        analysis_metric = data.get("analysis_metric")
        top_n = data.get("top_n", 5)
        start_date = pd.to_datetime(data.get("start_date"))
        end_date = pd.to_datetime(data.get("end_date"))

        # Normalize inputs
        selected_attribute, analysis_metric = normalize_inputs(
            selected_attribute, analysis_metric
        )

        df = get_session_data(session_id)
        if df is None:
            return jsonify(
                {
                    "error": "Session not found or expired. Please upload your data again."
                }
            ), 400

        # Prepare data for visualization
        df_plot = prepare_df_for_visual_plots(
            df,
            selected_attribute=selected_attribute,
            analysis_metric=analysis_metric,
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
        )

        # Close any existing plots
        plt.close("all")

        # Generate the plot
        fig = plot_final_frame(
            df=df_plot,
            top_n=top_n,
            analysis_metric=analysis_metric,
            selected_attribute=selected_attribute,
            start_date=start_date,
            end_date=end_date,
            period="M",
            days=30,
            image_cache=image_cache,
            error_logged=error_logged,
        )

        buf = BytesIO()
        fig.savefig(buf, format="jpeg", dpi=300, facecolor="#F0F0F0", edgecolor="none")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        filename = f"{selected_attribute}_{analysis_metric}_visual.jpg"

        return jsonify({"image": image_base64, "filename": filename}), 200

    except Exception as e:
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/generate_animation", methods=["POST"])
def generate_animation():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        selected_attribute = data.get("selected_attribute")
        analysis_metric = data.get("analysis_metric")
        top_n = data.get("top_n", 5)
        start_date = pd.to_datetime(data.get("start_date"))
        end_date = pd.to_datetime(data.get("end_date"))
        speed_for_bar_animation = data.get("speed_for_bar_animation", 28)
        days = data.get("days", 30)
        interp_steps = data.get("interp_steps", 14)
        period = data.get("period", "d")

        # Get DataFrame from Redis
        df = get_session_data(session_id)
        if df is None:
            return jsonify(
                {
                    "error": "Session not found or expired. Please upload your data again."
                }
            ), 400

        df_anim = prepare_df_for_visual_anims(
            df,
            selected_attribute=selected_attribute,
            analysis_metric=analysis_metric,
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
        )

        anim_bar_plot = create_bar_animation(
            df_anim,
            top_n,
            analysis_metric,
            selected_attribute,
            period,
            dpi,
            days,
            interp_steps,
            start_date,
            end_date,
        )

        # Save animation to a temporary file for download
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file_path = temp_file.name
            anim_bar_plot.save(
                temp_file_path,
                writer="ffmpeg",
                fps=speed_for_bar_animation,
                savefig_kwargs={"facecolor": "#F0F0F0"},
            )
            temp_file.seek(0)
            video_bytes = temp_file.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        filename = f"{selected_attribute}_{analysis_metric}_animation.mp4"

        return jsonify({"video": video_base64, "filename": filename}), 200

    except Exception as e:
        return jsonify({"error": f"Animation generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
