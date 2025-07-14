import base64
import gc
import json
import os
import tempfile
import time
import traceback
import uuid
import zipfile
from io import BytesIO

import duckdb
import matplotlib
import polars as pl
import psutil
from flask import Flask, jsonify, request

matplotlib.use("Agg")
import sys

# Ensure the modules directory is in the Python path for backend deployment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import matplotlib.pyplot as plt
import pandas as pd
from flask_cors import CORS

from modules.create_bar_animation import create_bar_animation
from modules.create_bar_plot import plot_final_frame
from modules.normalize_inputs import normalize_inputs
from modules.prepare_visuals import error_logged, image_cache

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "/tmp/spotify_sessions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_SESSIONS = 1  # only one user can upload data at a time due to resource constraints


def log_mem(msg):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    print(f"{msg} - Memory usage: {mem_mb:.2f} MB")


def cleanup_old_sessions(upload_dir=UPLOAD_DIR, max_age_seconds=2700):
    """Remove files older than 45min from the upload directory."""
    now = time.time()
    for fname in os.listdir(upload_dir):
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
            os.remove(fpath)


def insert_jsons_from_zip_to_duckdb(zip_path, session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    db_path = os.path.join(UPLOAD_DIR, f"spotify_session_{session_id}.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS spotify_data (
            Date TIMESTAMP,
            duration_ms BIGINT,
            track_name VARCHAR,
            artist_name VARCHAR,
            album_name VARCHAR,
            track_uri VARCHAR
        )
    """)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        json_file_names = [
            filename
            for filename in file_list
            if filename.endswith(".json")
            and "Audio" in filename
            and not filename.endswith("/")
        ]
        for json_file_name in json_file_names:
            try:
                with zip_ref.open(json_file_name) as json_file:
                    json_content = json_file.read()
                    json_data = json.loads(json_content.decode("utf-8"))
                    if json_data:
                        filtered_data = [
                            {
                                "Date": row.get("ts"),
                                "duration_ms": row.get("ms_played") / 60000
                                if row.get("ms_played")
                                else None,
                                "track_name": row.get("master_metadata_track_name"),
                                "artist_name": row.get(
                                    "master_metadata_album_artist_name"
                                ),
                                "album_name": row.get(
                                    "master_metadata_album_album_name"
                                ),
                                "track_uri": row.get("spotify_track_uri"),
                            }
                            for row in json_data
                            if row.get("ms_played", 0) > 30000
                        ]
                        if filtered_data:
                            df = pl.DataFrame(filtered_data)
                            # Convert Date to datetime
                            df = df.with_columns(
                                pl.col("Date").str.strptime(
                                    pl.Datetime, "%Y-%m-%dT%H:%M:%SZ", strict=False
                                )
                            )
                            df = df.drop_nulls()
                            con.execute("INSERT INTO spotify_data SELECT * FROM df")
            except Exception as e:
                print(f"Warning: Could not process {json_file_name}: {e}")
                continue

    min_date = con.execute("SELECT MIN(Date) FROM spotify_data").fetchone()[0]
    max_date = con.execute("SELECT MAX(Date) FROM spotify_data").fetchone()[0]
    con.close()
    return session_id, min_date, max_date


@app.route("/process", methods=["POST"])
def process_zip():
    cleanup_old_sessions()
    # Limit concurrent sessions
    session_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".duckdb")]
    if len(session_files) >= MAX_SESSIONS:
        return jsonify(
            {
                "error": "Server is busy. Too many users are generating visuals right now. Please try again in a few minutes."
            }
        ), 503
    print("Received /process request")
    log_mem("Start /process")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            uploaded_file.save(zip_path)
            log_mem("After file save")
            session_id, start_date_file, end_date_file = (
                insert_jsons_from_zip_to_duckdb(zip_path)
            )
            response_data = {
                "session_id": session_id,
                "data_min_date": str(start_date_file),
                "data_max_date": str(end_date_file),
            }
            return jsonify(response_data), 200

    except Exception as e:
        log_mem(f"Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


def query_user_duckdb(
    session_id, selected_attribute, analysis_metric, start_date, end_date, top_n
):
    db_path = os.path.join(UPLOAD_DIR, f"spotify_session_{session_id}.duckdb")
    if not os.path.exists(db_path):
        return None
    metric_expr = (
        "COUNT(*) as Streams"
        if analysis_metric == "Streams"
        else "SUM(duration_ms) as duration_ms"
    )
    order_by = "Streams" if analysis_metric == "Streams" else "duration_ms"

    if selected_attribute == "artist_name":
        select_cols = "artist_name, MIN(track_uri) as track_uri"
        group_by = "artist_name"
    elif selected_attribute == "track_name":
        select_cols = "track_name, artist_name, track_uri"
        group_by = "track_name, artist_name, track_uri"
    elif selected_attribute == "album_name":
        select_cols = "album_name, artist_name, MIN(track_uri) as track_uri"
        group_by = "album_name, artist_name"
    else:
        select_cols = f"{selected_attribute}, MIN(track_uri) as track_uri"
        group_by = selected_attribute

    end_date_inclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    query = f"""
        SELECT {select_cols}, {metric_expr}
        FROM spotify_data
        WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
        GROUP BY {group_by}
        ORDER BY {order_by} DESC
        LIMIT {top_n}
    """
    con = duckdb.connect(db_path)
    result_df = con.execute(query).df()
    con.close()
    return result_df


def query_user_duckdb_for_animation(
    session_id,
    selected_attribute,
    analysis_metric,
    start_date,
    end_date,
    filter_number=100,
):
    db_path = os.path.join(UPLOAD_DIR, f"spotify_session_{session_id}.duckdb")
    if not os.path.exists(db_path):
        return None

    if analysis_metric == "Streams":
        metric_expr = "COUNT(*) as Streams"
        cumsum_expr = "SUM(COUNT(*)) OVER (PARTITION BY {group_by} ORDER BY Date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as Cumulative_Streams"
        order_by = "Streams"
    elif analysis_metric == "duration_ms":
        metric_expr = "SUM(duration_ms) as duration_ms"
        cumsum_expr = "SUM(SUM(duration_ms)) OVER (PARTITION BY {group_by} ORDER BY Date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as Cumulative_duration_ms"
        order_by = "duration_ms"

    end_date_inclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    con = duckdb.connect(db_path)

    if selected_attribute == "artist_name":
        top_entities_query = f"""
            SELECT artist_name, COUNT(*) as Streams, SUM(duration_ms) as duration_ms
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            GROUP BY artist_name
            ORDER BY {order_by} DESC
            LIMIT {filter_number}
        """
        top_entities = [row[0] for row in con.execute(top_entities_query).fetchall()]
        group_by = "artist_name"
        query = f"""
            SELECT
                artist_name,
                Date,
                regexp_extract(MIN(track_uri), '[^:]+$', 0) as track_uri,
                {metric_expr},
                {cumsum_expr.format(group_by=group_by)}
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            AND artist_name IN ({",".join(["?"] * len(top_entities))})
            GROUP BY artist_name, Date
            ORDER BY artist_name, Date
        """
        result_df = con.execute(query, top_entities).df()

    elif selected_attribute == "track_name":
        top_entities_query = f"""
            SELECT track_uri, COUNT(*) as Streams, SUM(duration_ms) as duration_ms
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            GROUP BY track_uri
            ORDER BY {order_by} DESC
            LIMIT {filter_number}
        """
        top_entities = [row[0] for row in con.execute(top_entities_query).fetchall()]
        group_by = "track_uri"
        query = f"""
            SELECT
                track_name,
                artist_name,
                Date,
                MIN(track_uri) as track_uri,
                {metric_expr},
                {cumsum_expr.format(group_by=group_by)}
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            AND track_uri IN ({",".join(["?"] * len(top_entities))})
            GROUP BY track_name, track_uri, artist_name, Date
            ORDER BY track_name, Date
        """
        result_df = con.execute(query, top_entities).df()

    elif selected_attribute == "album_name":
        top_entities_query = f"""
            SELECT album_name, artist_name, COUNT(*) as Streams, SUM(duration_ms) as duration_ms
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            GROUP BY album_name, artist_name
            ORDER BY {order_by} DESC
            LIMIT {filter_number}
        """
        top_entities = [
            (row[0], row[1]) for row in con.execute(top_entities_query).fetchall()
        ]
        group_by = "album_name"
        query = f"""
            SELECT
                album_name,
                artist_name,
                Date,
                MIN(track_uri) as track_uri,
                {metric_expr},
                {cumsum_expr.format(group_by=group_by)}
            FROM spotify_data
            WHERE Date >= '{start_date}' AND Date < '{end_date_inclusive}'
            GROUP BY album_name, track_uri, artist_name, Date
            ORDER BY album_name, Date
        """
        result_df = con.execute(query).df()

    con.close()
    if selected_attribute == "album_name":
        top_entities_set = set(top_entities)
        result_df = result_df[
            result_df.apply(
                lambda row: (row["album_name"], row["artist_name"]) in top_entities_set,
                axis=1,
            )
        ]
    else:
        pass
    return result_df


@app.route("/generate_image", methods=["POST"])
def generate_image():
    cleanup_old_sessions()
    try:
        log_mem("Start /generate_image")
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

        df = query_user_duckdb(
            session_id, selected_attribute, analysis_metric, start_date, end_date, top_n
        )
        log_mem("After query_user_duckdb")
        if df is None or df.empty:
            return jsonify(
                {
                    "error": "Session expired. Please upload your data again to generate visuals."
                }
            ), 400
        log_mem("After prepare_df_for_visual_plots")

        plt.close("all")
        fig = plot_final_frame(
            df=df,
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
        log_mem("After plot_final_frame")

        buf = BytesIO()
        fig.savefig(buf, format="jpeg", dpi=91, facecolor="#F0F0F0", edgecolor="none")
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
        t0 = time.time()
        log_mem("Start /generate_animation")
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
        dpi = data.get("dpi", 10)
        figsize = data.get("figsize", (16, 21.2))

        t1 = time.time()
        print(f"Time to parse request data: {t1 - t0:.2f} seconds")
        df = query_user_duckdb_for_animation(
            session_id, selected_attribute, analysis_metric, start_date, end_date
        )
        # print(f"Query result for animation: {df.head(10)}")
        t2 = time.time()
        print(f"Time to query DuckDB for animation: {t2 - t1:.2f} seconds")
        log_mem("After query_user_duckdb_for_animation")
        if df is None:
            return jsonify(
                {
                    "error": "Session expired. Please upload your data again to generate visuals."
                }
            ), 400
        log_mem("After prepare_df_for_visual_anims")
        t3 = time.time()
        anim_bar_plot = create_bar_animation(
            df,
            top_n,
            analysis_metric,
            selected_attribute,
            period,
            dpi,
            days,
            interp_steps,
            start_date,
            end_date,
            figsize,
        )
        t4 = time.time()
        print(f"Frame generation (matplotlib) time: {t4 - t3:.2f} seconds")
        log_mem("After create_bar_animation")

        # Save animation to a temporary file for download
        import tempfile

        t5 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file_path = temp_file.name
            anim_bar_plot.save(
                temp_file_path,
                writer="ffmpeg",
                fps=speed_for_bar_animation,
                savefig_kwargs={"facecolor": "#F0F0F0"},
                # extra_args=["-preset", "veryfast"],
                extra_args=[
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "30",
                ],
            )
            log_mem("After anim_bar_plot.save (ffmpeg encoding done)")
            temp_file.seek(0)
            video_bytes = temp_file.read()
        t6 = time.time()
        print(f"Encoding (ffmpeg) time: {t6 - t5:.2f} seconds")
        print(f"Total animation time: {t6 - t3:.2f} seconds")
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        filename = f"{selected_attribute}_{analysis_metric}_animation.mp4"
        try:
            os.remove(temp_file_path)
        except Exception as cleanup_exc:
            print(f"Warning: Could not delete temp animation file: {cleanup_exc}")
        del anim_bar_plot
        image_cache.clear()
        plt.close("all")
        gc.collect()
        gc.collect()
        ffmpeg_procs = [
            p
            for p in psutil.process_iter(["name"])
            if p.info["name"] and "ffmpeg" in p.info["name"]
        ]
        print(
            f"FFMPEG processes running after cleanup: {len(ffmpeg_procs)}", flush=True
        )
        return jsonify({"video": video_base64, "filename": filename}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Animation generation failed: {str(e)}"}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)
