from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from services.ai_service import (
    query_deepseek,
    analyze_feedback,
    predict_retention_risk,
    analyze_engagement,
    answer_hr_question,
    screen_resumes,
)
from utils.resume_parser import parse_resume
from services.document_service import (
    clear_hr_documents,
    clear_insights,
    clear_resumes,
    get_insights,
    store_insight,
    store_text_in_chromadb,
    retrieve_relevant_resumes,
    delete_resume,
    delete_hr_document,
    list_resumes,
    list_hr_documents,
)


app = Flask(__name__)
CORS(app)
Compress(app)  # Enable response compression

document_bp = Blueprint("document", __name__)

# Apply rate limiting (200 requests per minute per IP)
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "HR Assistant API is running"}), 200


@app.route("/analyze-resume", methods=["POST"])
def analyze_resume_api():
    """
    API endpoint to analyze an uploaded resume for suitability.
    Fetches the most relevant resumes from ChromaDB based on the query.
    """
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve and rank the most relevant resumes
    relevant_resumes = retrieve_relevant_resumes(query)

    if not relevant_resumes:
        return jsonify({"error": "No matching resumes found"}), 200

    # Format resume data for AI analysis
    formatted_resumes = "\n\n".join(
        [
            f"Rank {i+1} (Score: {r['score']:.4f}):\n{r['text']}"
            for i, r in enumerate(relevant_resumes)
        ]
    )

    # Improved AI prompt to ensure keyword relevance
    prompt = (
        "Below are resumes ranked by relevance to your query. Focus only on candidates that match the keywords "
        "and provide a structured summary of their qualifications. Do NOT invent information:\n\n"
        f"Query: {query}\n\n{formatted_resumes}"
    )

    # Get AI-generated analysis
    ai_response = query_deepseek(prompt)

    return jsonify(
        {"query": query, "analysis": ai_response, "resumes": relevant_resumes}
    )


@app.route("/predict-retention", methods=["POST"])
def retention_prediction():
    """
    API endpoint to predict employee retention risk.
    """
    data = request.json
    employee_data = data.get("employee_data")

    if not employee_data:
        return jsonify({"error": "Missing employee_data"}), 400

    result = predict_retention_risk(employee_data)

    # Store retention analysis in hr_insights
    store_insight("retention", result)

    return jsonify(
        {"message": "Retention analysis stored successfully.", "result": result}
    )


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    """
    API endpoint to upload a resume, extract text, and store it in ChromaDB.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Extract text directly from the uploaded file without saving it
    resume_text = parse_resume(file.stream)

    if not resume_text:
        return jsonify({"error": "Failed to extract text from resume"}), 400

    metadata = {
        "filename": file.filename,
        "type": "resume",
    }

    # Store in ChromaDB
    store_text_in_chromadb(resume_text, metadata)

    return jsonify(
        {
            "message": "Resume uploaded and stored successfully",
            "resume_text": resume_text,
        }
    )


@app.route("/analyze-feedback", methods=["POST"])
def analyze_feedback_api():
    """
    API endpoint to analyze employee feedback and determine sentiment.
    """
    data = request.get_json()

    if "feedback_text" not in data:
        return jsonify({"error": "Missing feedback_text"}), 400

    feedback_text = data["feedback_text"]

    # Pass feedback to AI analysis
    result = analyze_feedback(feedback_text)

    # Store sentiment analysis in hr_insights
    store_insight("sentiment", result)

    return jsonify(
        {"message": "Sentiment analysis stored successfully.", "result": result}
    )


@app.route("/analyze-engagement", methods=["POST"])
def analyze_engagement_api():
    """
    API endpoint to analyze multiple feedback entries and provide engagement insights.
    """
    data = request.get_json()

    if "feedback_list" not in data:
        return jsonify({"error": "Missing feedback_list"}), 400

    feedback_list = "\n\n".join(
        data["feedback_list"]
    )  # Combine feedback into a single string

    # Analyze aggregated feedback
    result = analyze_engagement(feedback_list)

    # Store engagement analysis in hr_insights
    store_insight("engagement", result)

    return jsonify(
        {"message": "Engagement analysis stored successfully.", "result": result}
    )


@app.route("/get-insights", methods=["GET"])
def get_insights_api():
    """
    API endpoint to fetch HR insights.
    If a 'type' query parameter is provided, filter insights by type.
    Returns all insights if no type is specified.
    """
    insight_type = request.args.get("type")
    insights = get_insights(insight_type)
    return jsonify({"insights": insights})


@app.route("/ask-hr", methods=["POST"])
def ask_hr_api():
    """
    API endpoint for employees to ask HR-related questions.
    """
    data = request.get_json()

    if "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    question = data["question"]

    # Get answer from HR policies
    result = answer_hr_question(question)

    return jsonify(result)


@app.route("/screen-resumes", methods=["POST"])
def screen_resumes_api():
    """
    API endpoint for screening resumes based on a job description.
    """
    data = request.get_json()

    if "job_description" not in data or "resumes" not in data:
        return jsonify({"error": "Missing job description or resumes"}), 400

    job_description = data["job_description"]
    resumes = data["resumes"]  # List of resume texts

    # Get AI evaluation
    result = screen_resumes(job_description, resumes)

    return jsonify(result)


@document_bp.route("/list-resumes", methods=["GET"])
def list_resumes_endpoint():
    """
    API endpoint to list all stored resumes.
    """
    result = list_resumes()
    return jsonify(result)


@document_bp.route("/list-hr-documents", methods=["GET"])
def list_hr_documents_endpoint():
    """
    API endpoint to list all stored HR documents.
    """
    result = list_hr_documents()
    return jsonify(result)


@document_bp.route("/delete-resume", methods=["DELETE"])
def delete_resume_endpoint():
    """
    API endpoint to delete a resume by filename.
    """
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    result = delete_resume(filename)
    return jsonify(result)


@document_bp.route("/delete-hr-document", methods=["DELETE"])
def delete_hr_document_endpoint():
    """
    API endpoint to delete an HR document by filename.
    """
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    result = delete_hr_document(filename)
    return jsonify(result)


@document_bp.route("/upload-resumes", methods=["POST"])
def upload_resumes():
    """
    API endpoint to upload multiple resumes, extract text, and store in ChromaDB's resumes collection.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    uploaded_files = []
    for file in files:
        if file.filename == "":
            continue  # Skip empty files

        resume_text = parse_resume(file.stream)
        if not resume_text:
            continue  # Skip files with no extractable text

        metadata = {
            "filename": file.filename,
            "type": "resume",
        }

        # Store in ChromaDB (no collection_name argument)
        store_text_in_chromadb(resume_text, metadata)
        uploaded_files.append(file.filename)

    if not uploaded_files:
        return jsonify({"error": "No valid resumes processed"}), 400

    return jsonify(
        {"message": "Resumes uploaded successfully", "uploaded_files": uploaded_files}
    )


@document_bp.route("/upload-hr-documents", methods=["POST"])
def upload_hr_documents():
    """
    API endpoint to upload multiple HR documents and store in ChromaDB's HR documents collection.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    uploaded_files = []
    for file in files:
        if file.filename == "":
            continue  # Skip empty files

        document_text = parse_resume(file.stream)
        if not document_text:
            continue  # Skip empty or invalid files

        metadata = {
            "filename": file.filename,
            "type": "hr_document",
        }

        # Store in ChromaDB
        store_text_in_chromadb(document_text, metadata)
        uploaded_files.append(file.filename)

    if not uploaded_files:
        return jsonify({"error": "No valid HR documents processed"}), 400

    return jsonify(
        {
            "message": "HR documents uploaded successfully",
            "uploaded_files": uploaded_files,
        }
    )


@document_bp.route("/clear-insights", methods=["DELETE"])
def clear_insights_endpoint():
    """
    API endpoint to delete all insights.
    """
    result = clear_insights()
    return jsonify(result)


@document_bp.route("/clear-resumes", methods=["DELETE"])
def clear_resumes_endpoint():
    """
    API endpoint to delete all resumes.
    """
    result = clear_resumes()
    return jsonify(result)


@document_bp.route("/clear-hr-documents", methods=["DELETE"])
def clear_hr_documents_endpoint():
    """
    API endpoint to delete all HR documents.
    """
    result = clear_hr_documents()
    return jsonify(result)


# Register document_bp at the end, after defining all its routes
app.register_blueprint(document_bp, url_prefix="/documents")

if __name__ == "__main__":
    app.run(debug=True)
