# chatbot.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ---- PART 2 — Chatbot Class ----

class Chatbot:
    def __init__(self):
        self.history = []  # this IS the memory
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant named Jarvis."
        }

    def get_response(self, user_input):
        # add user message to memory
        self.history.append({
            "role": "user",
            "content": user_input
        })

        # send full history to Groq
        response = client.chat.completions.create( 
            model="llama-3.3-70b-versatile",
            messages=[self.system_prompt] + self.history
        )

        # extract reply
        reply = response.choices[0].message.content

        # save reply to memory too
        self.history.append({
            "role": "assistant",
            "content": reply
        })

        return reply
    
        # ---- PART 3 — Chat Loop ----

def chat_loop():
    bot = Chatbot()  # create one instance of our chatbot
    print("🤖 Jarvis is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")  # take user input

        if user_input.lower() == "quit":
            print("Jarvis: Goodbye!")
            break

        if user_input.strip() == "":
            print("Jarvis: Please say something!\n")
            continue

        response = bot.get_response(user_input)
        print(f"Jarvis: {response}\n")

# ---- PART 4 — Run ----

if __name__ == "__main__":
    chat_loop()