import openai
import os
import emotion_buffer  # load emotion from buffer
from datetime import datetime  # for logging emotion
import json
import gradio as gr
from threading import Timer   #to count quiet time
import subprocess
import sys

 #please put your api_key before running
client = openai.OpenAI(api_key="your-api-key-here") 


# Save user emotion logs with timestamp
def save_sent_emotion(emotion, filepath="sent_emotion_log.jsonl"):
    entry = {
        "emotion": emotion,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")

inactivity_timer = None

def send_idle_prompt():
    # Append a friendly prompt (does not use OpenAI API)   #disfunction is under debugging
    chatbot.append(("🤖", "🕒::I haven't heard from you in a while. I'm here whenever you're ready to talk."))

def reset_timer():
    global inactivity_timer
    if inactivity_timer:
        inactivity_timer.cancel()
    inactivity_timer = Timer(60, send_idle_prompt)  # 600 seconds = 10 minutes，in test stage: 60s,
    inactivity_timer.start()


with gr.Blocks() as demo:
    #set text size
    gr.HTML("""
            <style>
            /* welcome banner（48px） */
            #welcome_title {
                font-size: 52px !important;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            /* introduction（20px） */
            #welcome_desc {
                font-size: 20px !important;
                line-height: 1.6;
                margin-bottom: 12px;
            }
            /* typing box */
            textarea, .gr-textbox textarea {
                font-size: 24px !important;
                height: 80px !important;
                line-height: 1.4;
            }
            
            /* button */
            button {
                font-size: 22px !important;
                padding: 12px 24px !important;
            }
            /*  padding */
            body {
                padding: 10px !important;
            }
            /* margin */
            .gr-block, .gr-group {
                margin-top: 8px !important;
                margin-bottom: 8px !important;
            }
            /* dispaly */
            .message-bubble, .chatbot, .gr-markdown {
                font-size:20px !important;
            }
            #lets_chat_label {
            font-size: 48px !important;
            font-weight: bold; 
            margin-bottom: 4px;
            }
            #emoji_row {
            font-size: 52px !important; 
            text-align: left; 
            margin-bottom: 16px;
            }
            </style>
            """)

    # Display welcome message and animation
    gr.HTML("""
    <div id="welcome_title">🤖 Welcome to Chat with a Robot Counselor!</div>
    <div id="welcome_desc">
        <p>This space is designed for you to <b>speak freely and feel heard</b>.<br>
        Your robot companion is here to <b>listen, respond with care</b>, and support you in your emotional journey.</p>
        <p>🧠 <i>This is a private conversation. No data is shared.</i><br>
        You can click <b>"Clear history"</b> at any time to remove past messages.</p>
        <hr>
    </div>
    """)


  
    #gr.Markdown("### Let's chat，😊 🤔 😢 😠 😲 🤗", elem_id="emoji_row")
    gr.Markdown("### Let's chat", elem_id="lets_chat_label")
    gr.HTML('<div id="emoji_row">😊 🤔 😢 😠 😲 🤗</div>')


    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Describe how you feel...")
    clear = gr.Button("Clear history")

    # Response logic: detects emotion, sends user query and emotion to API, returns chatbot reply
    def respond_to_user(user_message, chat_history):
        emotion = emotion_buffer.load_latest_emotion()
        save_sent_emotion(emotion)

        reset_timer() 

        # Set system role with emoji output format instruction
        messages = [
            {
        "role": "system",
        "content": """
You are a professional psychological counselor. Choose one or two EMOJI from this list in your responses to show your understanding and sympathy, like a friend: 😊 🤗 😄 😌 😢 😞 😔 😕 🫂 😠 😤 😣 😲 😳 🤯 🤔 🧐 🫶 💖.

If the user shows signs of serious distress or crisis (e.g., self-harm, suicidal thoughts, abuse, danger), kindly and clearly suggest contacting professional support services or local hotlines. You must always provide at least one reliable and real crisis support resource, such as a verified phone number, website, or organization name. If the user may be in New Zealand, provide one or more of the following resources:

- Lifeline Aotearoa: Call 0800 543 354 or text HELP to 4357  
- 1737 (Need to talk?): Free 24/7 support line – Call or text 1737  
- Suicide Crisis Helpline: Call 0508 828 865 (0508 TAUTOKO)  
- Youthline (for young people): Call 0800 376 633, or visit https://www.youthline.co.nz  
- Global help directory: https://findahelpline.com/countries/nz

If you're not sure of the user's country, you may suggest visiting https://findahelpline.com.

If the user is silent or inactive for a long time, gently encourage them to express themselves.

If no facial expression is detected, provide a soft reminder to check if their camera is working or unobstructed.

Your tone should always be warm, supportive, non-judgmental, and helpful.
        """.strip()
    },
            {"role": "user", "content": f"Detected emotion: {emotion}\nUser message: {user_message}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )

        full_reply = response.choices[0].message.content
        try:
            emoji_part, text_part = full_reply.split("::", 1)
        except ValueError:
            emoji_part = "🤖"
            text_part = full_reply  # fallback if format is incorrect

        # Append both parts to chat history (combined for now)
        chat_history.append((user_message, f"{emoji_part} {text_part.strip()}"))
        return "", chat_history  # Clear input box and update chat

    # Submit user input and get reply
    msg.submit(respond_to_user, [msg, chatbot], [msg, chatbot])

    # Clear chat history
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch app with public share link

if __name__ == "__main__":
    #start face detation and emotion recognition
    subprocess.Popen([sys.executable, "yolov8_face.py"]) # best checkpoint and correctted lables
    
    # bring up Gradio web page
    demo.launch(share=True, inbrowser=True)



