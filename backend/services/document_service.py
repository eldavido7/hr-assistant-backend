import time
import uuid
from chromadb import PersistentClient
import re
from utils.resume_parser import parse_resume
import logging

logging.basicConfig(level=logging.INFO)

# Set a consistent path for ChromaDB storage
CHROMA_DB_PATH = "data/chroma_db"  # Ensure this directory exists

# Initialize ChromaDB client
client = PersistentClient(path=CHROMA_DB_PATH)

# Create or access collections
hr_collection = client.get_or_create_collection(name="hr_documents")  # For HR policies
resume_collection = client.get_or_create_collection(name="resumes")  # For resumes
hr_insights_collection = client.get_or_create_collection(
    name="hr_insights"
)  # For HR insights


def retrieve_relevant_text(query):
    """
    Retrieves relevant text from stored HR documents based on the query.
    Returns empty string if no documents are available or if collection is empty.
    """
    try:
        # Check if collection has any documents
        collection_count = hr_collection.count()
        print(f"HR Collection document count: {collection_count}")

        if collection_count == 0:
            logging.warning("HR collection is empty. No documents have been uploaded.")
            return ""

        # Query the collection
        results = hr_collection.query(query_texts=[query], n_results=3)

        # Debug: Log what we got back
        print(f"Query results structure: {results.keys()}")
        print(f"Documents found: {len(results.get('documents', [[]])[0])}")

        # Extract relevant chunks from the results
        relevant_texts = [
            doc
            for doc_list in results.get("documents", [])
            for doc in doc_list
            if doc and doc.strip() and doc.strip() not in ["N/A", "N/A N/A"]
        ]

        if not relevant_texts:
            logging.warning(f"No relevant HR policies found for query: '{query}'")
            return ""

        # Join and clean the texts
        combined_text = "\n\n".join(relevant_texts)

        # Additional validation - check if we got real content
        if (
            len(combined_text.strip()) < 10
        ):  # Less than 10 chars is probably not real content
            logging.warning(f"Retrieved text too short: '{combined_text}'")
            return ""

        print(
            f"Successfully retrieved {len(combined_text)} characters of relevant text"
        )
        return combined_text

    except Exception as e:
        logging.error(f"Error retrieving relevant text: {e}", exc_info=True)
        return ""


def store_text_in_chromadb(text, metadata):
    """
    Stores extracted resume text in ChromaDB for later retrieval.
    Ensures job_role is always stored properly.
    """
    metadata["timestamp"] = time.time()
    doc_id = metadata.get("filename")

    try:
        resume_collection.add(ids=[doc_id], documents=[text], metadatas=[metadata])
        logging.info(
            f"Stored document with ID: {doc_id} in ChromaDB (resumes collection)"
        )
    except Exception as e:
        logging.error(f"Failed to store document '{doc_id}' in ChromaDB: {str(e)}")


def retrieve_relevant_resumes(query):
    """
    Searches stored resumes in ChromaDB and returns relevant matches ranked by relevance.
    Filters out results that do not contain at least one important keyword.
    """
    results = resume_collection.query(
        query_texts=[query], n_results=10  # Fetch more results before filtering
    )

    if not results["documents"]:
        return []

    # Extract keywords from query
    keywords = extract_keywords(query)

    retrieved_resumes = []
    for i in range(len(results["documents"][0])):
        resume_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        score = results["distances"][0][i]

        # Only include resumes that contain at least one keyword from the query
        if any(keyword in resume_text.lower() for keyword in keywords):
            retrieved_resumes.append(
                {"text": resume_text, "metadata": metadata, "score": score}
            )

    # Sort resumes by relevance score (lower = more relevant)
    retrieved_resumes.sort(key=lambda x: x["score"])

    return retrieved_resumes


def extract_keywords(query):
    """
    Extracts important keywords from a query using a simple regex-based approach.
    """
    # Convert query to lowercase and split by non-alphanumeric characters
    words = re.findall(r"\b\w+\b", query.lower())

    # Define common words to ignore (stopwords)
    stopwords = {
        "find",
        "me",
        "a",
        "an",
        "the",
        "with",
        "years",
        "experience",
        "developer",
        "engineer",
        "for",
        "of",
    }

    # Filter out stopwords
    keywords = [word for word in words if word not in stopwords]

    return keywords


def save_hr_document(file_path, file_name):
    """
    Extracts text from an HR document using the same parsing logic as resumes
    and stores it in ChromaDB.
    """
    document_text = parse_resume(file_path)  # Use the same resume parser

    if not document_text.strip():
        return {"error": "The document is empty or could not be processed."}

    # Store document text in ChromaDB
    hr_collection.add(
        documents=[document_text], metadatas=[{"filename": file_name}], ids=[file_name]
    )

    return {"message": "HR document uploaded successfully."}


def save_bulk_hr_documents(files):
    """
    Processes and stores multiple HR documents in ChromaDB.
    """
    uploaded_files = []

    for file in files:
        if file.filename == "":
            logging.warning("Skipped an empty file with no filename.")
            continue  # Skip empty files

        document_text = parse_resume(file.stream)
        if not document_text.strip():
            logging.warning(
                f"Skipped file '{file.filename}' as it contains no valid text."
            )
            continue  # Skip empty or invalid files

        # Use the existing save_hr_document function
        result = save_hr_document(file.stream, file.filename)
        if "message" in result:
            uploaded_files.append(file.filename)

    if not uploaded_files:
        logging.error("No valid HR documents processed.")
        return {"error": "No valid HR documents processed"}

    return {
        "message": "HR documents uploaded successfully",
        "uploaded_files": uploaded_files,
    }


def delete_resume(filename):
    """
    Deletes a resume from ChromaDB by its filename.
    """
    try:
        resume_collection.delete(ids=[filename])
        logging.info(f"Deleted resume with ID: {filename} from ChromaDB")
        return {"message": f"Resume '{filename}' deleted successfully."}
    except Exception as e:
        logging.error(f"Failed to delete resume '{filename}': {str(e)}")
        return {"error": f"Failed to delete resume: {str(e)}"}


def delete_hr_document(filename):
    """
    Deletes an HR document from ChromaDB by its filename.
    """
    try:
        hr_collection.delete(ids=[filename])
        logging.info(f"Deleted HR document with ID: {filename} from ChromaDB")
        return {"message": f"HR document '{filename}' deleted successfully."}
    except Exception as e:
        logging.error(f"Failed to delete HR document '{filename}': {str(e)}")
        return {"error": f"Failed to delete HR document: {str(e)}"}


def list_resumes():
    """
    Lists all stored resume filenames in ChromaDB.
    """
    all_resumes = resume_collection.get()

    if not all_resumes["ids"]:
        return {"resumes": []}

    return {"resumes": all_resumes["ids"]}


def list_hr_documents():
    """
    Lists all stored HR document filenames in ChromaDB.
    """
    all_docs = hr_collection.get()

    if not all_docs["ids"]:
        return {"hr_documents": []}

    return {"hr_documents": all_docs["ids"]}


def store_insight(insight_type, data):
    """
    Stores HR insights in ChromaDB.
    :param insight_type: Type of insight (e.g., "engagement", "sentiment", "retention").
    :param data: The analyzed result to store.
    """
    insight_id = f"{insight_type}_{uuid.uuid4()}"  # Unique ID
    try:
        hr_insights_collection.add(
            ids=[insight_id],
            documents=[data],
            metadatas=[{"type": insight_type, "timestamp": time.time()}],
        )
        logging.info(f"Stored {insight_type} insight in ChromaDB with ID: {insight_id}")
    except Exception as e:
        logging.error(f"Failed to store {insight_type} insight: {str(e)}")


def get_insights(insight_type=None):
    """
    Retrieves stored HR insights of a specific type or all types.
    :param insight_type: Type of insight to fetch (optional).
    :param limit: Number of recent insights to return per type.
    """
    if insight_type:
        # For specific type, query only that type
        query_texts = [insight_type]
    else:
        # For all insights, use an empty query to match everything
        query_texts = [""]

    results = hr_insights_collection.query(query_texts=query_texts, n_results=100)

    insights = [
        {"data": doc, "metadata": meta}
        for doc_list, meta_list in zip(
            results.get("documents", []), results.get("metadatas", [])
        )
        for doc, meta in zip(doc_list, meta_list)
    ]

    if not insights:
        logging.info(
            f"No {'insights' if not insight_type else f'{insight_type} insights'} found."
        )
        return []

    # If a specific type was requested, filter results to match that type
    if insight_type:
        insights = [
            insight
            for insight in insights
            if insight.get("metadata", {}).get("type") == insight_type
        ]

    return insights


def clear_insights():
    """
    Clears all HR insights from the ChromaDB collection.
    """
    try:
        # Use a condition that matches all documents
        hr_insights_collection.delete(where={"document_id": {"$ne": ""}})
        logging.info("Cleared all insights from ChromaDB")
        return {"message": "All insights cleared successfully."}
    except Exception as e:
        logging.error(f"Failed to clear HR insights: {str(e)}")
        return {"error": f"Failed to clear HR insights: {str(e)}"}


def clear_resumes():
    """
    Deletes all resumes from the resumes collection.
    """
    try:
        # Use a condition that matches all documents
        resume_collection.delete(where={"document_id": {"$ne": ""}})
        logging.info("Cleared all resumes from ChromaDB")
        return {"message": "All resumes deleted successfully."}
    except Exception as e:
        logging.error(f"Failed to clear resumes: {str(e)}")
        return {"error": f"Failed to clear resumes: {str(e)}"}


def clear_hr_documents():
    """
    Deletes all HR documents from the hr_documents collection.
    """
    try:
        # Use a condition that matches all documents
        hr_collection.delete(where={"document_id": {"$ne": ""}})
        logging.info("Cleared all HR documents from ChromaDB")
        return {"message": "All HR documents deleted successfully."}
    except Exception as e:
        logging.error(f"Failed to clear HR documents: {str(e)}")
        return {"error": f"Failed to clear HR documents: {str(e)}"}
