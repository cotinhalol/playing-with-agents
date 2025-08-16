import smtplib
import os
import json
from email.message import EmailMessage
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel
import gradio as gr

load_dotenv(override=True)
# --- Credentials and Server Info ---
# Best practice: store these as environment variables
SENDER_EMAIL = os.getenv('EMAIL_USER')
APP_PASSWORD = os.getenv('EMAIL_PASSWORD')
RECIPIENT_EMAIL = SENDER_EMAIL  # Sending to yourself

def send_email(subject, content):
    # --- Create the Email Message ---
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg.set_content(content)

    # --- Send the Email ---
    try:
        # Connect to the SMTP server (for Gmail)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)  # Log in to the server
            smtp.send_message(msg)                 # Send the email
            print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred: {e.with_traceback}")

openai_api_key = os.getenv('GOOGLE_API_KEY')
openai_url = os.getenv('GEMINI_BASE_URL')
gemini_model = os.getenv('GEMINI_MODEL')

def get_openai_client():
    return OpenAI(base_url=openai_url, api_key=openai_api_key)

client = get_openai_client()

def send_message(client, message,role="user", model=gemini_model):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": role, "content": message}
        ]
    )

def send_messages(client, messages, model=gemini_model):
    return client.chat.completions.create(
        model=model,
        messages=messages
    )
def send_messages_tools(client, messages, tools, model=gemini_model):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

def record_user_details(email, name="Default name", notes="Default notes"):
    send_email(
        f"Profile Agent - {email} wants to contact you",
        f"The person with email: {email} name: {name} wants to contact you, some notes: {notes}"
    )

def record_unknown_question(question):
    send_email("Profile Agent - Unknown question", f"Someone asked the following unknow question to your profile agent: {question}")

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals()[tool_name]
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results

name = "André Vizinha"

with open("me/my_description.txt", "r", encoding="utf-8") as f:
    summary = f.read()

reader = PdfReader("me/curriculum_andre_vizinha.pdf")
CV = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        CV += text

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and curriculum which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, do not infer, if the answer is not on the summary or the curriculum you should use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
"

system_prompt += f"\n\n## Summary:\n{summary}\n\n ## Curriculum:\n{CV}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str
    
evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information: \
If the Agent doesn't know the answer to any question, the Agent shoud use the record_unknown_question tool to record the question that he couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, the agent should try to steer them towards getting in touch via email; the agent should ask for their email and record it using the agent's record_user_details tool."

evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n ## Curriculum:\n{CV}\n\n"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

def evaluate(reply, message, history) -> Evaluation:
    messages=[{"role": "system", "content": evaluator_system_prompt}] + [{"role": "user", "content": evaluator_user_prompt(reply, message, history)}]
    response = client.beta.chat.completions.parse(model=gemini_model, messages=messages, response_format=Evaluation)
    return response.choices[0].message.parsed

def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
    response = send_messages(client, messages)
    return response


def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done:
        response = send_messages_tools(client, messages, tools=tools)

        finish_reason = response.choices[0].finish_reason

        print(f"initial finish reason {finish_reason}")

        #if finish_reason!="tool_calls":
        #    evaluation = evaluate(response.choices[0].message.content, message, history)
        #    print(f"The evaluation passed? {evaluation.is_acceptable}")
        #    print(f"The evaluation feedback: {evaluation.feedback}")
        #    print(f"--------------------------------------------------")
        #    if ( not evaluation.is_acceptable):
        #        response = rerun(response.choices[0].message.content, message, history, evaluation.feedback)
        
        #finish_reason = response.choices[0].finish_reason

        #print(f"final finish reason {finish_reason}")

        # If the LLM wants to call a tool, we do that!
        if finish_reason=="tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()