import json
import re
from flask import jsonify
import requests
import os
from dotenv import load_dotenv
from services.document_service import retrieve_relevant_text
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR)

load_dotenv()  # Load environment variables from .env

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)  # Store API key in environment variables

DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the .env file")

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

if not PHONE_NUMBER_ID or not WHATSAPP_ACCESS_TOKEN:
    raise ValueError(
        "Missing WHATSAPP_PHONE_NUMBER_ID or WHATSAPP_ACCESS_TOKEN in environment variables."
    )


# Configure token limits
MAX_OUTPUT_TOKENS = 2000  # Optimized for Render free tier
MAX_CONTEXT_TOKENS = 12000  # Slightly under max for efficiency and to avoid errors


def query_deepseek(prompt):
    """
    Sends a prompt to DeepSeek AI and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek/deepseek-chat-v3.1:free",  # Primary model
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        # "context_length": MAX_CONTEXT_TOKENS,
    }

    try:
        print(f"Sending prompt to DeepSeek: {prompt[:100]}...")
        response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
        response.raise_for_status()

        result = response.json()

        if (
            "choices" in result
            and result["choices"]
            and "message" in result["choices"][0]
            and "content" in result["choices"][0]["message"]
            and result["choices"][0]["message"]["content"].strip()
        ):
            content = result["choices"][0]["message"]["content"]
            print(f"DeepSeek raw response content: {content[:200]}...")

            # Wrap response into JSON format expected by the app
            return json.dumps({"answer": content})
        else:
            logging.error(
                "DeepSeek API returned an empty or invalid response structure."
            )
            logging.error(f"Full response: {result}")
            return json.dumps(
                {
                    "answer": "I apologize, but I couldn't generate a proper response. Can you send that message again?"
                }
            )
    except requests.RequestException as e:
        logging.error(f"DeepSeek API request failed: {e}")
        return json.dumps(
            {"answer": f"I'm having technical difficulties right now: {str(e)}"}
        )
    except Exception as e:
        logging.error(f"Unexpected error querying DeepSeek: {e}")
        return json.dumps({"answer": f"An unexpected error occurred: {str(e)}"})


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
    Answers questions using uploaded documents.
    Handles greetings and general conversation dynamically.
    """
    # Retrieve relevant content from HR documents
    relevant_text = retrieve_relevant_text(question)

    # DEBUG: Check what's being retrieved
    print(f"Retrieved text length: {len(relevant_text) if relevant_text else 0}")
    print(
        f"Retrieved text preview: {relevant_text[:200] if relevant_text else 'EMPTY'}"
    )

    # Handle case where no documents are available
    if not relevant_text or relevant_text.strip() in ["", "N/A", "N/A N/A"]:
        # Check if it's a greeting or casual conversation
        greetings = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        gratitude = ["thank", "thanks", "appreciate"]

        question_lower = question.lower().strip()

        if any(greeting in question_lower for greeting in greetings):
            return json.dumps(
                {
                    "answer": "Hello! I'm here to help you with HR-related questions. However, it seems no HR documents have been uploaded yet. Please contact your administrator to upload the necessary documents."
                }
            )

        if any(thank in question_lower for thank in gratitude):
            return json.dumps(
                {
                    "answer": "You're welcome! If you have any other questions, feel free to ask."
                }
            )

        # No documents available for actual questions
        return json.dumps(
            {
                "answer": "I apologize, but I don't have access to any HR documents at the moment. Please contact your administrator to upload the necessary HR policies and information."
            }
        )

    # Documents are available - proceed with normal flow
    prompt = f"""You are a helpful HR assistant. Answer the question using ONLY the information provided below.

Question: {question}

Available Information:
{relevant_text}

Instructions:
1. If the question is a greeting (hello, hi, good morning):
   - Respond warmly and professionally
   - Invite them to ask HR-related questions
   
2. If the question expresses gratitude (thank you, thanks):
   - Acknowledge warmly
   - Offer further assistance
   
3. If the question can be answered with the available information:
   - Provide a clear, detailed answer
   - Use only facts from the provided information
   - Be professional but friendly
   
4. If the question CANNOT be answered with the available information:
   - Politely explain that the specific information isn't available
   - Do NOT mention "documents" or "sources"
   - List the general topics you CAN help with based on the available information
   - Invite them to rephrase or ask about those topics

CRITICAL: Respond ONLY with valid JSON in this exact format:
{{
  "answer": "your response here"
}}

Do NOT include any text before or after the JSON. Do NOT include labels like "Classification:" or "Response:".
"""

    response = query_deepseek(prompt)

    # Additional validation
    if not response or response.strip() == "":
        return json.dumps(
            {
                "answer": "I apologize, but I encountered an error processing your question. Please try again."
            }
        )

    return response


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


def process_deepseek_response(response):
    """
    Extracts the answer text from DeepSeek's response.

    Handles multiple formats:
    - Direct dict: {"answer": "text"}
    - JSON string: '{"answer": "text"}'
    - Code blocks: ```json\n{"answer": "text"}\n```
    - Nested JSON: {"answer": "{\"answer\": \"text\"}"}
    - Empty/malformed responses

    Args:
        response: Response from DeepSeek API (dict, str, or None)

    Returns:
        str: Cleaned answer text or error message
    """
    EMPTY_RESPONSE_MSG = "I apologize, but I couldn't generate a proper response. Can you send that message again?"

    # Handle None/empty
    if not response:
        return EMPTY_RESPONSE_MSG

    try:
        # Convert to string for uniform processing
        if isinstance(response, dict):
            response = json.dumps(response)

        response_str = str(response).strip()

        if not response_str:
            return EMPTY_RESPONSE_MSG

        # Remove code block wrappers (```json ... ``` or ``` ... ```)
        if response_str.startswith("```"):
            lines = response_str.split("\n")
            # Remove first line (```json or ```) and last line (```)
            response_str = "\n".join(lines[1:-1]).strip()

        # Try to parse as JSON (handles both direct JSON and nested JSON strings)
        answer = response_str
        max_depth = 3  # Prevent infinite loops on circular structures

        for _ in range(max_depth):
            # If it looks like JSON, try to parse it
            if answer.startswith("{") and answer.endswith("}"):
                try:
                    parsed = json.loads(answer)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        answer = parsed["answer"]
                        # If answer is still a dict/list, convert back to string
                        if isinstance(answer, (dict, list)):
                            answer = json.dumps(answer)
                        continue
                    else:
                        # JSON parsed but no "answer" key, use the JSON string
                        break
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain text
                    break
            else:
                # Doesn't look like JSON, we're done
                break

        # Final cleanup
        answer = str(answer).strip()

        # Remove markdown bold/italic formatting
        answer = answer.replace("**", "").replace("*", "")

        # Remove DeepSeek 3.1 model artifacts (appears at end of responses)
        artifacts_to_remove = [
            "<｜begin▁of▁sentence｜>",
            "<|begin_of_sentence|>",
            "<｜end▁of▁sentence｜>",
            "<|end_of_sentence|>",
        ]

        for artifact in artifacts_to_remove:
            if answer.endswith(artifact):
                answer = answer[: -len(artifact)].strip()
            # Also check if it appears anywhere in the text
            answer = answer.replace(artifact, "").strip()

        # Check if we ended up with empty content
        if not answer or answer in ['""', "''", "{}", "[]"]:
            return EMPTY_RESPONSE_MSG

        return answer

    except Exception as e:
        logger.error(f"Error processing DeepSeek response: {e}", exc_info=True)
        logger.debug(f"Problematic response: {str(response)[:500]}")
        return EMPTY_RESPONSE_MSG


TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def handle_telegram_request(data):
    """
    Handles incoming Telegram messages and sends responses.
    """
    chat_id = data["message"]["chat"]["id"]
    question = data["message"].get("text", "").strip()

    if not question:
        return jsonify({"status": "No question provided"}), 200

    try:
        # Get answer from HR policies
        result = answer_hr_question(question)
        print("Raw AI Response (Telegram):", result)

        # Ensure result is properly extracted
        response_text = process_deepseek_response(result)
        print("Processed answer (Telegram):", response_text)

        # Extra cleaning to remove any remaining markdown or JSON formatting
        # Clean up any remaining code blocks
        if "```" in response_text:
            # Remove all code block formatting
            response_text = re.sub(r"```(?:json|python|)\n", "", response_text)
            response_text = response_text.replace("```", "")

        # Extra step to clean any remaining JSON formatting
        if response_text.strip().startswith("{") and response_text.strip().endswith(
            "}"
        ):
            try:
                json_response = json.loads(response_text)
                if "answer" in json_response:
                    response_text = json_response["answer"]
            except:
                pass

        # Final check for empty response
        if not response_text or response_text.strip() == "":
            response_text = "I apologize, but I couldn't generate a proper response. How else can I help you?"

        # Send response to Telegram
        send_response = requests.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": response_text},
            timeout=10,
        )

        if send_response.status_code != 200:
            print(f"Failed to send message to Telegram: {send_response.text}")
            return jsonify({"status": "Message processed but sending failed"}), 200

        return jsonify({"status": "Message processed"}), 200

    except Exception as e:
        print(f"Exception in handle_telegram_request: {str(e)}")
        return jsonify({"status": "Error but acknowledged"}), 200


def handle_whatsapp_request(data):
    """
    Process incoming WhatsApp message and reply back using the HR AI system.
    """
    try:
        entry = data["entry"][0]
        changes = entry.get("changes", [])
        if not changes:
            return jsonify({"status": "no_changes"}), 200

        message_data = changes[0].get("value", {}).get("messages", [])
        if not message_data:
            return jsonify({"status": "no_messages"}), 200

        message = message_data[0]
        sender_phone_number = message.get("from")
        message_text = message.get("text", {}).get("body")

        if not sender_phone_number or not message_text:
            return jsonify({"status": "invalid_message"}), 200

        # Create or reuse session (even without history, this is useful)
        session = get_or_create_whatsapp_session(sender_phone_number)
        print(f"Session info for {sender_phone_number}: {session}")

        # Send question to DeepSeek
        result = answer_hr_question(message_text)

        print("Raw AI Response:", result)

        processed_answer = process_deepseek_response(result)
        print("Processed answer:", processed_answer)

        # Clean up any remaining code blocks
        if "```" in processed_answer:
            processed_answer = re.sub(r"```(?:json|python|)\n", "", processed_answer)
            processed_answer = processed_answer.replace("```", "")

        # Add this extra step to clean any remaining JSON formatting (same as in Telegram handler)
        if processed_answer.strip().startswith(
            "{"
        ) and processed_answer.strip().endswith("}"):
            try:
                json_response = json.loads(processed_answer)
                if "answer" in json_response:
                    processed_answer = json_response["answer"]
            except:
                pass

        if not processed_answer.strip():
            processed_answer = "I apologize, but I couldn't generate a proper response. Can you send that message again?"

        # Send response back to WhatsApp
        send_whatsapp_message(sender_phone_number, processed_answer)

        return jsonify({"status": "message_processed"}), 200

    except Exception as e:
        print(f"Error handling WhatsApp message: {e}")
        return jsonify({"error": "Failed to process message"}), 500


def send_whatsapp_message(phone_number, message):
    """
    Send a reply message back to the user on WhatsApp.
    """
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"body": message},
    }

    response = requests.post(url, headers=headers, json=payload)
    print(f"WhatsApp send message response: {response.status_code}, {response.text}")

    if response.status_code >= 400:
        print(f"Failed to send WhatsApp message to {phone_number}: {response.text}")


from datetime import datetime, timedelta

# Global session store (in-memory for now)
whatsapp_sessions = {}


def get_or_create_whatsapp_session(phone_number):
    now = datetime.now(timezone.utc)

    # Cleanup expired sessions (24-hour window)
    for number, session in list(whatsapp_sessions.items()):
        if (now - session["last_message_time"]) > timedelta(hours=24):
            del whatsapp_sessions[number]

    if phone_number not in whatsapp_sessions:
        whatsapp_sessions[phone_number] = {
            "created_at": now,
            "last_message_time": now,
            "user_type": "new",
        }
        print(f"New session created for {phone_number}")
    else:
        whatsapp_sessions[phone_number]["last_message_time"] = now
        whatsapp_sessions[phone_number]["user_type"] = "returning"

    return whatsapp_sessions[phone_number]
