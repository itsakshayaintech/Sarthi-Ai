import os
import tempfile
import json
import secrets
import sqlite3
import sys
from functools import wraps
from getpass import getpass

import joblib
import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash


# =========================================================
# APPLICATION SETUP
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
import tempfile

DATABASE_DIR = os.path.join(tempfile.gettempdir(), "sarthi_database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "sahayak.sqlite3")
app.config.update(
    SECRET_KEY=os.environ.get("SAHAYAK_SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=20 * 1024 * 1024,
)


def get_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    with get_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'worker')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                final_svi REAL NOT NULL,
                risk_category TEXT NOT NULL,
                assessment_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


initialize_database()

def ensure_worker_account():
    worker_name = os.environ.get("SAHAYAK_WORKER_NAME")
    worker_email = os.environ.get("SAHAYAK_WORKER_EMAIL")
    worker_password = os.environ.get("SAHAYAK_WORKER_PASSWORD")

    if not worker_name or not worker_email or not worker_password:
        return

    if len(worker_password) < 8:
        print("Worker password must contain at least 8 characters.")
        return

    worker_email = worker_email.strip().lower()

    with get_database() as connection:
        existing_worker = connection.execute(
            "SELECT id FROM users WHERE email = ? AND role = 'worker'",
            (worker_email,)
        ).fetchone()

        if existing_worker:
            connection.execute(
                """
                UPDATE users
                SET name = ?, password_hash = ?
                WHERE id = ?
                """,
                (
                    worker_name,
                    generate_password_hash(worker_password),
                    existing_worker["id"]
                )
            )
        else:
            connection.execute(
                """
                INSERT INTO users
                (name, email, password_hash, role)
                VALUES (?, ?, ?, 'worker')
                """,
                (
                    worker_name,
                    worker_email,
                    generate_password_hash(worker_password)
                )
            )

def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token}


def validate_csrf():
    submitted = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    if not submitted or not secrets.compare_digest(
        submitted, session.get("csrf_token", "")
    ):
        abort(400, description="Your session token expired. Refresh and try again.")


def require_role(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("role") not in roles:
                if request.is_json or request.path.startswith("/analyze-audio"):
                    return jsonify({"error": "Please sign in to continue."}), 401
                return redirect(url_for("login"))
            if request.method == "POST":
                validate_csrf()
            return view(*args, **kwargs)

        return wrapped

    return decorator


# =========================================================
# LOAD INDIVIDUAL MODELS
# =========================================================

print("\nLoading Sarthi-AI models...")


# ---------------------------------------------------------
# Indicator Model
# ---------------------------------------------------------

indicator_data = joblib.load(
    os.path.join(
        MODEL_DIR,
        "indicator_model.pkl"
    )
)


# ---------------------------------------------------------
# Voice Model
# ---------------------------------------------------------

voice_data = joblib.load(
    os.path.join(
        MODEL_DIR,
        "voice_model.pkl"
    )
)


# ---------------------------------------------------------
# Context Model
# ---------------------------------------------------------

context_data = joblib.load(
    os.path.join(
        MODEL_DIR,
        "context_model.pkl"
    )
)


# ---------------------------------------------------------
# Text Model
# ---------------------------------------------------------

text_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "text_model.pkl"
    )
)


# =========================================================
# EXTRACT MODELS AND FEATURES
# =========================================================

indicator_model = indicator_data["model"]
indicator_features = indicator_data["features"]


voice_model = voice_data["model"]
voice_features = voice_data["features"]


context_model = context_data["model"]
context_features = context_data["features"]


print("\nModels loaded successfully.")

print(
    "Indicator Features:",
    indicator_features
)

print(
    "Voice Features:",
    voice_features
)

print(
    "Context Features:",
    context_features
)


# =========================================================
# RISK CATEGORY
# =========================================================

def get_risk_category(svi):

    svi = float(svi)

    if svi <= 25:
        return "Low"

    elif svi <= 50:
        return "Moderate"

    elif svi <= 75:
        return "High"

    else:
        return "Critical"


# =========================================================
# SUPPORT RECOMMENDATION ENGINE
# =========================================================

def get_support_recommendation(risk):

    recommendations = {

        "Low": [
            "Provide general support and information.",
            "Offer appropriate counselling or welfare guidance.",
            "Follow-up may be considered where appropriate."
        ],

        "Moderate": [
            "Consider counselling or support referral.",
            "Review reported concerns with an authorized professional.",
            "Consider appropriate follow-up."
        ],

        "High": [
            "Priority professional review is recommended.",
            "Conduct an appropriate safety assessment.",
            "Consider suitable support pathways."
        ],

        "Critical": [
            "Immediate professional human review is required.",
            "Conduct an urgent safety assessment.",
            "Consider appropriate emergency and support pathways."
        ]
    }

    return recommendations.get(
        risk,
        []
    )


# =========================================================
# FUSION ENGINE
# =========================================================

def calculate_final_svi(scores):

    """
    Combines the available modality scores.

    If the trained fusion model exists:
        Individual SVI scores
                ↓
          Fusion Model
                ↓
           Final SVI

    If fusion_model.pkl does not exist yet:
        Uses average as a temporary fallback.

    This allows the Flask application to run before
    the fusion model has been trained.
    """

    if not scores:

        return None


    # -----------------------------------------------------
    # Extract individual modality scores
    # -----------------------------------------------------

    indicator_svi = scores.get(
        "indicator_svi"
    )

    text_svi = scores.get(
        "text_svi"
    )

    voice_svi = scores.get(
        "voice_svi"
    )

    context_svi = scores.get(
        "context_svi"
    )


    # -----------------------------------------------------
    # Try trained fusion model
    # -----------------------------------------------------

    fusion_model_path = os.path.join(
        MODEL_DIR,
        "fusion_model.pkl"
    )


    if os.path.exists(fusion_model_path):

        try:

            from inference.fusion_inference import (
                predict_fusion
            )

            final_svi = predict_fusion(

                indicator_svi=indicator_svi,

                text_svi=text_svi,

                voice_svi=voice_svi,

                context_svi=context_svi
            )

            return round(
                float(
                    np.clip(
                        final_svi,
                        0,
                        100
                    )
                ),
                2
            )

        except Exception as error:

            print(
                "Fusion model error:",
                error
            )

            print(
                "Using temporary average fallback."
            )


    # -----------------------------------------------------
    # TEMPORARY FALLBACK
    # -----------------------------------------------------

    valid_scores = [

        score

        for score in [
            indicator_svi,
            text_svi,
            voice_svi,
            context_svi
        ]

        if score is not None
    ]


    if not valid_scores:

        return None


    final_svi = np.mean(
        valid_scores
    )


    return round(
        float(
            np.clip(
                final_svi,
                0,
                100
            )
        ),
        2
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    if session.get("role") == "worker":
        return redirect(url_for("worker_dashboard"))
    if session.get("role") == "user":
        return redirect(url_for("user_dashboard"))
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        validate_csrf()
        name = request.form.get("name", "").strip()
        
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter your name and email, and use a password with at least 8 characters.")
            return render_template("auth.html", mode="register"), 400
        try:
            with get_database() as connection:
                cursor = connection.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'user')",
                    (name, email, generate_password_hash(password)),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            flash("An account with that email already exists.")
            return render_template("auth.html", mode="register"), 409
        session.clear()
        session.update(user_id=user_id, name=name, role="user")
        return redirect(url_for("user_dashboard"))
    return render_template("auth.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    return authenticate("user")


@app.route("/worker/login", methods=["GET", "POST"])
def worker_login():
    return authenticate("worker")


def authenticate(role):
    if request.method == "POST":
        validate_csrf()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_database() as connection:
            user = connection.execute(
                "SELECT id, name, password_hash FROM users WHERE email = ? AND role = ?",
                (email, role),
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.update(user_id=user["id"], name=user["name"], role=role)
            destination = "worker_dashboard" if role == "worker" else "user_dashboard"
            return redirect(url_for(destination))
        flash("Email or password is incorrect.")
    return render_template("auth.html", mode="worker_login" if role == "worker" else "login")


@app.route("/logout", methods=["POST"])
def logout():
    validate_csrf()
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
@require_role("user")
def user_dashboard():
    with get_database() as connection:
        assessments = connection.execute(
            "SELECT id, final_svi, risk_category, created_at FROM assessments WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (session["user_id"],),
        ).fetchall()
    return render_template("user_dashboard.html", assessments=assessments)


@app.route("/worker/dashboard")
@require_role("worker")
def worker_dashboard():
    with get_database() as connection:
        totals = connection.execute(
            "SELECT COUNT(*) AS total, ROUND(AVG(final_svi), 1) AS average FROM assessments"
        ).fetchone()
        categories = connection.execute(
            "SELECT risk_category, COUNT(*) AS count FROM assessments GROUP BY risk_category"
        ).fetchall()
        recent = connection.execute(
            "SELECT id, final_svi, risk_category, assessment_json, created_at FROM assessments ORDER BY id DESC LIMIT 20"
        ).fetchall()
    cases = []
    for assessment in recent:
        item = dict(assessment)
        item["scores"] = json.loads(item.pop("assessment_json"))
        cases.append(item)
    return render_template(
        "worker_dashboard.html",
        totals=totals,
        categories=categories,
        cases=cases,
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    fusion_model_exists = os.path.exists(
        os.path.join(
            MODEL_DIR,
            "fusion_model.pkl"
        )
    )

    return jsonify({

        "status": "online",

        "models": {

            "indicator": True,

            "text": True,

            "voice": True,

            "context": True,

            "fusion": fusion_model_exists
        }
    })


# =========================================================
# AUDIO ANALYSIS
# =========================================================

@app.route(
    "/analyze-audio",
    methods=["POST"]
)
@require_role("user")
def analyze_audio():

    if request.form.get("ai_consent") != "true":
        return jsonify({"error": "Consent is required before audio analysis."}), 400

    # -----------------------------------------------------
    # Check audio upload
    # -----------------------------------------------------

    if "audio" not in request.files:

        return jsonify({
            "error":
                "No audio file uploaded."
        }), 400


    audio_file = request.files[
        "audio"
    ]


    if audio_file.filename == "":

        return jsonify({
            "error":
                "No audio file selected."
        }), 400


    temp_path = None


    try:

        # -------------------------------------------------
        # Save uploaded audio temporarily
        # -------------------------------------------------

        suffix = os.path.splitext(
            audio_file.filename
        )[1]


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            audio_file.save(
                temp_file.name
            )

            temp_path = temp_file.name


        # -------------------------------------------------
        # SPEECH TO TEXT
        # -------------------------------------------------

        recognizer = sr.Recognizer()


        with sr.AudioFile(
            temp_path
        ) as source:

            audio_data = recognizer.record(
                source
            )


        try:

            transcript = recognizer.recognize_google(
                audio_data,
                language="en-IN"
            )


        except sr.UnknownValueError:

            transcript = ""


        except sr.RequestError:

            return jsonify({

                "error":
                    "Speech recognition service is unavailable."

            }), 503


        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        y, sample_rate = librosa.load(
            temp_path,
            sr=None
        )


        # -------------------------------------------------
        # AUDIO DURATION
        # -------------------------------------------------

        duration = librosa.get_duration(
            y=y,
            sr=sample_rate
        )


        # -------------------------------------------------
        # ENERGY
        # -------------------------------------------------

        rms = librosa.feature.rms(
            y=y
        )[0]


        normalized_energy = float(
            np.mean(rms)
        )


        # -------------------------------------------------
        # PITCH
        # -------------------------------------------------

        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sample_rate
        )


        pitch_values = []


        for frame in range(
            pitches.shape[1]
        ):

            index = magnitudes[
                :,
                frame
            ].argmax()


            pitch = pitches[
                index,
                frame
            ]


            if pitch > 0:

                pitch_values.append(
                    float(pitch)
                )


        if pitch_values:

            pitch_variability_hz = float(
                np.std(
                    pitch_values
                )
            )

        else:

            pitch_variability_hz = 0.0


        # -------------------------------------------------
        # SPEECH RATE
        # -------------------------------------------------

        words = transcript.split()

        word_count = len(
            words
        )


        if duration > 0:

            speech_rate = (
                word_count /
                duration
            )

        else:

            speech_rate = 0.0


        # -------------------------------------------------
        # PAUSE ESTIMATE
        # -------------------------------------------------

        avg_pause = 0.0


        if len(y) > 0:

            silence = (
                np.abs(y) < 0.01
            )


            avg_pause = float(
                np.mean(silence)
                * duration
            )


        # -------------------------------------------------
        # HESITATION WORDS
        # -------------------------------------------------

        hesitation_words = [
            "um",
            "uh",
            "hmm",
            "er"
        ]


        hesitation_count = sum(

            word.lower().strip(
                ".,!?"
            )

            in hesitation_words

            for word in words
        )


        if word_count > 0:

            hesitation_ratio = (
                hesitation_count /
                word_count
            )

        else:

            hesitation_ratio = 0.0


        # -------------------------------------------------
        # VOICE FEATURES
        # -------------------------------------------------

        voice_features_result = {

            "speech_rate_words_per_sec":
                round(
                    float(
                        speech_rate
                    ),
                    4
                ),

            "avg_pause_seconds":
                round(
                    float(
                        avg_pause
                    ),
                    4
                ),

            "pitch_variability_hz":
                round(
                    float(
                        pitch_variability_hz
                    ),
                    4
                ),

            "normalized_energy":
                round(
                    float(
                        normalized_energy
                    ),
                    4
                ),

            "hesitation_ratio":
                round(
                    float(
                        hesitation_ratio
                    ),
                    4
                ),

            "speech_duration_sec":
                round(
                    float(
                        duration
                    ),
                    4
                )
        }


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "transcript":
                transcript,

            "voice_features":
                voice_features_result
        })


    except Exception as error:

        return jsonify({

            "error":
                str(error)

        }), 500


    finally:

        # -------------------------------------------------
        # Remove temporary audio file
        # -------------------------------------------------

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass


# =========================================================
# MAIN ASSESSMENT API
# =========================================================

@app.route(
    "/assess",
    methods=["POST"]
)
@require_role("user")
def assess():

    # -----------------------------------------------------
    # READ JSON
    # -----------------------------------------------------

    data = request.get_json()


    if not data:

        return jsonify({

            "error":
                "JSON input is required."

        }), 400


    # -----------------------------------------------------
    # CONSENT
    # -----------------------------------------------------

    consent = data.get(
        "ai_consent",
        False
    )


    # -----------------------------------------------------
    # NO AI CONSENT
    # -----------------------------------------------------

    if not consent:

        return jsonify({

            "ai_analysis":
                False,

            "message":
                "AI analysis is disabled because consent was not provided.",

            "next_step":
                "Use Human Assessment Support Form.",

            "human_assessment_fields": [

                "immediate_safety_concern",

                "reported_threat",

                "support_requested",

                "vulnerability_concern",

                "follow_up_possible"
            ]
        })


    # =====================================================
    # AI CONSENT GIVEN
    # =====================================================

    scores = {}


    # -----------------------------------------------------
    # INDICATOR MODEL
    # -----------------------------------------------------

    if "indicators" in data:

        indicators = data[
            "indicators"
        ]


        row = pd.DataFrame([

            {

                feature:
                    indicators.get(
                        feature,
                        0
                    )

                for feature
                in indicator_features
            }

        ])


        score = indicator_model.predict(
            row
        )[0]


        scores[
            "indicator_svi"
        ] = round(

            float(
                np.clip(
                    score,
                    0,
                    100
                )
            ),

            2
        )


    # -----------------------------------------------------
    # TEXT MODEL
    # -----------------------------------------------------

    if data.get("text"):

        text_input = str(
            data["text"]
        ).strip()


        if text_input:

            score = text_model.predict([
                text_input
            ])[0]


            scores[
                "text_svi"
            ] = round(

                float(
                    np.clip(
                        score,
                        0,
                        100
                    )
                ),

                2
            )


    # -----------------------------------------------------
    # VOICE MODEL
    # -----------------------------------------------------

    if "voice_features" in data:

        voice = data[
            "voice_features"
        ]


        row = pd.DataFrame([

            {

                feature:
                    voice.get(
                        feature,
                        0
                    )

                for feature
                in voice_features
            }

        ])


        score = voice_model.predict(
            row
        )[0]


        scores[
            "voice_svi"
        ] = round(

            float(
                np.clip(
                    score,
                    0,
                    100
                )
            ),

            2
        )


    # -----------------------------------------------------
    # CONTEXT MODEL
    # -----------------------------------------------------

    if "case_context" in data:

        context = data[
            "case_context"
        ]


        row = pd.DataFrame([

            {

                feature:
                    context.get(
                        feature,
                        None
                    )

                for feature
                in context_features
            }

        ])


        score = context_model.predict(
            row
        )[0]


        scores[
            "context_svi"
        ] = round(

            float(
                np.clip(
                    score,
                    0,
                    100
                )
            ),

            2
        )


    # -----------------------------------------------------
    # CHECK WHETHER ANY MODEL PRODUCED A SCORE
    # -----------------------------------------------------

    if not scores:

        return jsonify({

            "error":
                "No assessment data was provided."

        }), 400


    # -----------------------------------------------------
    # FINAL SVI
    # -----------------------------------------------------

    final_svi = calculate_final_svi(
        scores
    )


    if final_svi is None:

        return jsonify({

            "error":
                "Unable to calculate final SVI."

        }), 500


    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    risk_category = get_risk_category(
        final_svi
    )


    # -----------------------------------------------------
    # SUPPORT RECOMMENDATION
    # -----------------------------------------------------

    support_recommendation = (
        get_support_recommendation(
            risk_category
        )
    )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    result = {

        "ai_analysis":
            True,

        "assessment_scores": {

            "indicator_svi":
                scores.get(
                    "indicator_svi"
                ),

            "text_svi":
                scores.get(
                    "text_svi"
                ),

            "voice_svi":
                scores.get(
                    "voice_svi"
                ),

            "context_svi":
                scores.get(
                    "context_svi"
                )
        },

        "final_svi":
            final_svi,

        "risk_category":
            risk_category,

        "support_recommendation":
            support_recommendation,

        "important_note":
            "This system provides AI-assisted decision "
            "support only. Final decisions must be made "
            "by authorized human professionals."
    }
    with get_database() as connection:
        connection.execute(
            "INSERT INTO assessments (user_id, final_svi, risk_category, assessment_json) VALUES (?, ?, ?, ?)",
            (
                session["user_id"],
                final_svi,
                risk_category,
                json.dumps(result["assessment_scores"]),
            ),
        )
    return jsonify(result)


# =========================================================
# RUN APPLICATION
# =========================================================

# =========================================================
# WORKER ACCOUNT SETUP
# =========================================================

def ensure_worker_account():
    worker_name = os.environ.get("SAHAYAK_WORKER_NAME")
    worker_email = os.environ.get("SAHAYAK_WORKER_EMAIL")
    worker_password = os.environ.get("SAHAYAK_WORKER_PASSWORD")

    # If worker credentials are not configured, do nothing.
    if not worker_name or not worker_email or not worker_password:
        print("Worker environment variables are not configured.")
        return

    if len(worker_password) < 8:
        print("Worker password must contain at least 8 characters.")
        return

    worker_email = worker_email.strip().lower()

    with get_database() as connection:
        existing_worker = connection.execute(
            "SELECT id FROM users WHERE email = ? AND role = 'worker'",
            (worker_email,),
        ).fetchone()

        if existing_worker:
            connection.execute(
                """
                UPDATE users
                SET name = ?, password_hash = ?
                WHERE id = ?
                """,
                (
                    worker_name,
                    generate_password_hash(worker_password),
                    existing_worker["id"],
                ),
            )

            print("Helpline worker account updated.")

        else:
            connection.execute(
                """
                INSERT INTO users
                (name, email, password_hash, role)
                VALUES (?, ?, ?, 'worker')
                """,
                (
                    worker_name,
                    worker_email,
                    generate_password_hash(worker_password),
                ),
            )

            print("Helpline worker account created.")


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "create-worker":

        worker_name = input("Worker name: ").strip()
        worker_email = input("Worker email: ").strip().lower()
        worker_password = getpass(
            "Worker password (8+ characters): "
        )

        if len(worker_password) < 8:
            raise SystemExit(
                "Password must have at least 8 characters."
            )

        try:
            with get_database() as connection:
                connection.execute(
                    """
                    INSERT INTO users
                    (name, email, password_hash, role)
                    VALUES (?, ?, ?, 'worker')
                    """,
                    (
                        worker_name,
                        worker_email,
                        generate_password_hash(worker_password),
                    ),
                )

            print("Helpline worker account created.")

        except sqlite3.IntegrityError:
            raise SystemExit(
                "An account with that email already exists."
            )

    elif len(sys.argv) > 1 and sys.argv[1] == "reset-user-password":

        user_email = input("User email: ").strip().lower()
        user_password = getpass("New password (8+ characters): ")

        if len(user_password) < 8:
            raise SystemExit(
                "Password must have at least 8 characters."
            )

        with get_database() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET password_hash = ?
                WHERE email = ? AND role = 'user'
                """,
                (generate_password_hash(user_password), user_email),
            )

        if cursor.rowcount:
            print("User password updated.")
        else:
            raise SystemExit("No user account found for that email.")

    else:

        # Automatically create/update worker on Vercel
        ensure_worker_account()

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=os.environ.get("FLASK_DEBUG") == "1"
        )