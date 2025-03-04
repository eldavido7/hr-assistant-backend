import json
import re
from flask import jsonify
import requests
import os
from dotenv import load_dotenv
from services.document_service import retrieve_relevant_text
import logging
from datetime import datetime, timezone

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


def query_deepseek(prompt):
    """
    Sends a prompt to DeepSeek AI and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },  # Generic system message
            {"role": "user", "content": prompt},  # Send actual prompt as user message
        ],
    }

    try:
        print(f"Sending prompt to DeepSeek: {prompt[:100]}...")
        response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)

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

            # Try to format the response as JSON if it's not already
            if not (content.strip().startswith("{") and content.strip().endswith("}")):
                try:
                    # If it's plain text, wrap it in our expected JSON format
                    return json.dumps({"answer": content})
                except Exception as e:
                    logging.error(f"Error formatting response: {e}")
                    return json.dumps(
                        {"answer": content}
                    )  # Return anyway, don't silently fail

            return content
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


# def answer_hr_question(question):
#     """
#     Answers HR-related questions using uploaded HR documents.
#     Handles greetings and general conversation dynamically.
#     """
#     # Retrieve relevant content from HR documents
#     relevant_text = retrieve_relevant_text(question)

#     prompt = f"""
#     You are an AI HR assistant. Answer the following question using only the provided HR policy documents.

#     Question: {question}

#     Relevant HR Policy Documents:
#     {relevant_text}

#     If the answer is not found in the provided documents, politely explain that the answer is not available in the current HR policies, and suggest the user clarify the question or ask something else related to documents.

#     Additionally, if the question is a greeting, polite message, or general chat (e.g., "hello", "thank you", "how are you?", "good morning"):
#     - Respond appropriately in a friendly, professional manner.
#     - If the message is just a greeting, keep it short and engaging (e.g., "Hello! How can I assist you today?").
#     - If the message expresses gratitude (e.g., "thank you"), acknowledge it in a warm way (e.g., "You're very welcome! Let me know if you need anything else.").
#     - Keep responses polite and slightly humorous when appropriate, but always professional.

#     Provide a structured response in exactly this JSON format:
#     {{
#       "answer": "<your detailed answer here>"
#     }}
#     """

#     return query_deepseek(prompt)


def answer_hr_question(question):
    """
    Answers questions using uploaded documents.
    Handles greetings and general conversation dynamically.
    """
    # Retrieve relevant content from HR documents
    relevant_text = retrieve_relevant_text(question)

    prompt = f"""
    You are an AI assistant. Answer the following question using only the provided documents.

    Question: {question}

    Relevant Documents:
    {relevant_text}

    If the answer is not found in the provided documents, politely explain that the answer is not available in the uploaded files, and suggest the user clarify the question or ask something else related to documents.

    Additionally, if the question is a greeting, polite message, or general chat (e.g., "hello", "thank you", "how are you?", "good morning"):
    - Respond appropriately in a friendly, professional manner.
    - If the message is just a greeting, keep it short and engaging (e.g., "Hello! How can I assist you today?").
    - If the message expresses gratitude (e.g., "thank you"), acknowledge it in a warm way (e.g., "You're very welcome! Let me know if you need anything else.").
    - Keep responses polite and slightly humorous when appropriate, but always professional.

    Provide a structured response in exactly this JSON format:
    {{
      "answer": "<your detailed answer here>"
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


def process_deepseek_response(response):
    """
    Extracts the actual answer from DeepSeek's response, handling multiple formats and edge cases.
    """
    # Handle empty responses
    if not response:
        return "I apologize, but I couldn't generate a proper response. Can you send that message again?"

    print(
        f"Processing response type: {type(response)}, content: {str(response)[:200]}..."
    )

    # If we have an empty answer, return a default message
    if response == '{"answer": ""}' or response == {"answer": ""}:
        return "I apologize, but I couldn't generate a proper response. Can you send that message again?"

    # Direct check for simple JSON object with just an answer key (your specific case)
    if isinstance(response, dict) and len(response) == 1 and "answer" in response:
        answer_text = response["answer"]
        # Remove any asterisks from the response
        if isinstance(answer_text, str):
            answer_text = answer_text.replace("*", "")
            return answer_text.strip()

        # NEW CHECK: Handle the case where the response is a string representation of JSON with an answer key
        if (
            isinstance(response, str)
            and response.strip().startswith("{")
            and response.strip().endswith("}")
        ):
            try:
                json_obj = json.loads(response)
                if "answer" in json_obj:
                    answer_text = json_obj["answer"]
                    if isinstance(answer_text, str):
                        answer_text = answer_text.replace("*", "")
                        return answer_text.strip()
            except json.JSONDecodeError:
                pass

    # Handle code block format (```json {...} ```)
    if isinstance(response, str) and "```json" in response:
        try:
            # Extract content between ```json and ```
            json_content = response.split("```json")[1].split("```")[0].strip()
            parsed_json = json.loads(json_content)
            if "answer" in parsed_json:
                answer_text = parsed_json["answer"]
                # Remove any asterisks
                if isinstance(answer_text, str):
                    answer_text = answer_text.replace("*", "")
                    return answer_text.strip()
        except:
            pass

    try:
        # If response is already a dictionary
        if isinstance(response, dict):
            if "answer" in response:
                answer_content = response["answer"]

                # Check for code blocks in the answer string
                if isinstance(answer_content, str) and "```json" in answer_content:
                    try:
                        # Extract content between ```json and ```
                        json_content = (
                            answer_content.split("```json")[1].split("```")[0].strip()
                        )
                        parsed_json = json.loads(json_content)
                        if "answer" in parsed_json:
                            answer_text = parsed_json["answer"]
                            # Remove any asterisks
                            if isinstance(answer_text, str):
                                answer_text = answer_text.replace("*", "")
                                return answer_text.strip()
                    except:
                        pass

                if isinstance(answer_content, str):
                    # Try to parse the answer as JSON if it looks like JSON
                    if answer_content.strip().startswith(
                        "{"
                    ) and answer_content.strip().endswith("}"):
                        try:
                            inner_dict = json.loads(answer_content)
                            if "answer" in inner_dict:
                                answer_text = inner_dict["answer"]
                                # Remove any asterisks
                                if isinstance(answer_text, str):
                                    answer_text = answer_text.replace("*", "")
                                    return answer_text.strip()
                        except:
                            pass
                    # Otherwise return it directly with asterisks removed
                    answer_text = answer_content
                    if isinstance(answer_text, str):
                        answer_text = answer_text.replace("*", "")
                        return answer_text.strip()
                elif isinstance(answer_content, dict) and "answer" in answer_content:
                    answer_text = answer_content["answer"]
                    # Remove any asterisks
                    if isinstance(answer_text, str):
                        answer_text = answer_text.replace("**", "")
                        return answer_text.strip()

            # If we get here, return the string representation with asterisks removed
            result = str(response)
            return result.replace("*", "").strip()

        # If response is a string
        if isinstance(response, str):
            # Check for empty content
            if not response.strip():
                return "I apologize, but I couldn't generate a proper response. Can you send that message again?"

            # Try to parse it as JSON
            try:
                # Check if it's already a simple JSON string with just an answer key
                if response.strip().startswith(
                    '{"answer":'
                ) and response.strip().endswith("}"):
                    response_dict = json.loads(response)
                    if "answer" in response_dict:
                        answer_content = response_dict["answer"]
                        # Extra check for empty answer
                        if not answer_content or answer_content.strip() == "":
                            return "I apologize, but I couldn't generate a proper response. Can you send that message again?"
                        # Remove any asterisks
                        if isinstance(answer_content, str):
                            answer_content = answer_content.replace("*", "")
                        return answer_content.strip()

                # Otherwise proceed with normal parsing
                response_dict = json.loads(response)

                # If it has an answer key, process that
                if "answer" in response_dict:
                    answer_content = response_dict["answer"]

                    # Extra check for empty answer
                    if not answer_content or (
                        isinstance(answer_content, str) and answer_content.strip() == ""
                    ):
                        return "I apologize, but I couldn't generate a proper response. Can you send that message again?"

                    # Check for code blocks in the answer string
                    if isinstance(answer_content, str) and "```json" in answer_content:
                        try:
                            # Extract content between ```json and ```
                            json_content = (
                                answer_content.split("```json")[1]
                                .split("```")[0]
                                .strip()
                            )
                            parsed_json = json.loads(json_content)
                            if "answer" in parsed_json:
                                answer_text = parsed_json["answer"]
                                # Remove any asterisks
                                if isinstance(answer_text, str):
                                    answer_text = answer_text.replace("*", "")
                                    return answer_text.strip()
                        except:
                            pass

                    # If the answer is a string that looks like JSON
                    if isinstance(
                        answer_content, str
                    ) and answer_content.strip().startswith("{"):
                        try:
                            inner_dict = json.loads(answer_content)
                            if "answer" in inner_dict:
                                answer_text = inner_dict["answer"]
                                # Remove any asterisks
                                if isinstance(answer_text, str):
                                    answer_text = answer_text.replace("*", "")
                                    return answer_text.strip()
                        except:
                            # If it fails to parse as JSON, return the string directly with asterisks removed
                            if isinstance(answer_content, str):
                                answer_content = answer_content.replace("*", "")
                            return answer_content.strip()
                    # If answer is already a dict
                    elif (
                        isinstance(answer_content, dict) and "answer" in answer_content
                    ):
                        answer_text = answer_content["answer"]
                        # Remove any asterisks
                        if isinstance(answer_text, str):
                            answer_text = answer_text.replace("*", "")
                            return answer_text.strip()
                    # Otherwise return the answer string directly with asterisks removed
                    else:
                        if isinstance(answer_content, str):
                            answer_content = answer_content.replace("*", "")
                        return answer_content.strip()

                # Fallback: return any content we can find with asterisks removed
                result = str(response_dict)
                return result.replace("*", "").strip()

            except json.JSONDecodeError:
                # If not JSON, return the raw string if it's not empty, with asterisks removed
                if response.strip():
                    return response.strip().replace("*", "")

    except Exception as e:
        print(f"Error processing DeepSeek response: {e}")

    return "I'm not sure how to answer that."


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
