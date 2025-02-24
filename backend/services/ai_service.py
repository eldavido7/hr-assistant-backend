import requests
import os
from dotenv import load_dotenv
from services.document_service import retrieve_relevant_text
import logging

logging.basicConfig(level=logging.ERROR)

load_dotenv()  # Load environment variables from .env

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)  # Store API key in environment variables

DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def query_deepseek(prompt):
    """
    Sends a prompt to DeepSeek AI and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek/deepseek-chat:free",  # Use the appropriate DeepSeek AI model
        "messages": [{"role": "system", "content": prompt}],
    }

    response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)

    try:
        response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        result = response.json()
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        else:
            logging.error("DeepSeek API returned an empty response.")
            return {"error": "No valid response from AI."}
    except requests.RequestException as e:
        logging.error(f"DeepSeek API request failed: {e}")
        return {"error": f"DeepSeek API request failed: {str(e)}"}


def analyze_resume(resume_text, job_role):
    """
    Analyzes a resume using DeepSeek AI to determine its suitability for a job role.
    """
    prompt = f"""
    You are an AI HR assistant reviewing resumes for the position of {job_role}. 
    The resume text is provided below. Extract relevant details such as skills, experience, education, 
    and provide an evaluation of how well this candidate matches the role.

    Resume:
    {resume_text}

    Provide a structured response in the following format:
    {{
      "match_score": "<Score out of 100>",
      "key_skills": ["skill1", "skill2", "skill3"],
      "experience_summary": "<Brief summary of candidate's experience>",
      "education": "<Highest degree and university>",
      "strengths": ["strength1", "strength2"],
      "weaknesses": ["weakness1", "weakness2"],
      "suggestions": "Recommendations to improve the resume or fit the role better"
    }}
    """

    return query_deepseek(prompt)


def predict_retention_risk(employee_data):
    """
    Predicts retention risk based on employee history and engagement data.
    """
    prompt = f"""
    You are an AI HR assistant. Based on the following employee data, predict their retention risk.
    
    Employee Data:
    {employee_data}
    
    Provide a structured response with risk level (low, medium, high) and reasons.
    """
    return query_deepseek(prompt)


def analyze_feedback(feedback_text):
    """
    Analyzes employee feedback using DeepSeek AI to determine sentiment and key topics.
    """
    prompt = f"""
    You are an AI HR assistant analyzing employee feedback. 

    Feedback: 
    {feedback_text}

    Analyze the sentiment (Positive, Neutral, or Negative) and extract key concerns or topics mentioned. 
    Provide a structured response in the following format:
    {{
      "sentiment": "<Positive, Neutral, or Negative>",
      "key_topics": ["topic1", "topic2"],
      "summary": "<Brief summary of the employee's concern>",
      "recommendations": "Suggestions for HR to address this feedback."
    }}
    """

    return query_deepseek(prompt)


def analyze_engagement(feedback_list):
    """
    Aggregates employee feedback to detect engagement trends.
    """
    if not feedback_list:
        return {"error": "No feedback data provided."}

    prompt = f"""
    You are an AI HR assistant analyzing employee engagement based on feedback trends.

    Below is a collection of employee feedback:
    {feedback_list}

    Identify recurring topics, overall sentiment distribution, and key trends.
    Provide a structured response in this format:
    {{
      "overall_sentiment_distribution": {{
        "positive": "<Percentage of positive feedback>",
        "neutral": "<Percentage of neutral feedback>",
        "negative": "<Percentage of negative feedback>"
      }},
      "top_recurring_topics": ["topic1", "topic2"],
      "summary": "<Brief summary of key engagement trends>",
      "recommendations": "Suggestions for improving employee engagement."
    }}
    """

    return query_deepseek(prompt)


def answer_hr_question(question):
    """
    Answers HR-related questions using uploaded HR documents.
    Handles greetings and general conversation dynamically while keeping responses relevant.
    """
    # Retrieve relevant content from HR documents
    relevant_text = retrieve_relevant_text(question)

    prompt = f"""
    You are an AI HR assistant. Answer the following question using only the provided HR policy documents.

    Question: {question}

    Relevant HR Policy Documents:
    {relevant_text}

    If the answer is not found in the provided documents, respond with:
    "I couldn't find an answer in the available HR policies, please ask a question related to it, thank you."

    Additionally, if the question is a greeting, polite message, or general chat (e.g., "hello", "thank you", "how are you?", "good morning"):
    - Respond appropriately in a friendly, professional manner.
    - If the message is just a greeting, keep it short and engaging (e.g., "Hello! How can I assist you today?").
    - If the message expresses gratitude (e.g., "thank you"), acknowledge it in a warm way (e.g., "You're very welcome! Let me know if you need anything else.").
    - Keep responses polite and slightly humorous when appropriate, but always professional.

    Provide a structured response:
    {{
      "answer": "<AI-generated answer based on HR documents or appropriate general response>"
    }}
    """

    return query_deepseek(prompt)


def screen_resumes(job_description, resumes):
    """
    Screens resumes against a job description using DeepSeek AI.
    """
    prompt = f"""
    You are an AI-powered HR assistant evaluating resumes for a job opening.

    Job Description:
    {job_description}

    Below are the candidate resumes:
    {resumes}

    Evaluate each resume based on its relevance to the job description.
    Rank the resumes from most suitable to least suitable and provide a short reason for each ranking.

    Format your response as:
    [
      {{"name": "<Candidate Name>", "score": <Score out of 10>, "reason": "<Why this candidate was ranked this way>"}}
    ]
    """

    return query_deepseek(prompt)
