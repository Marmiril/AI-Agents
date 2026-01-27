import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model 
llm = OllamaLLM(model="mistral")

# Initialize Memory (LangChain v1.0+)
chat_history = ChatMessageHistory()

# Initialize Text-to-Speech Engine.
engine = pyttsx3.init()

engine.setProperty("rate", 160) # Adjust speaking speed. About 200 is standard

# Speech Recognition
recognizer = sr.Recognizer()

# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen
def listen():
    with sr.Microphone() as source:
        print("\n Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("I didn`t undestand a thing honey... ")
        return""
    except sr.RequestError:
        print("Speech Reconginiton Service Unavailable")
        return""


# AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to Process AI Responses
def run_chain(question):
    # Retrieve past chat history manually
    chat_history_text="\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

    # Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    chat_history.add_user_message(question) 
    chat_history.add_ai_message(response)

    return response

# Main loop
speak("Hello! how can I help you?")

while True: 
    query = listen()
    if "exit" in query or "stop" in query:
        speak("Goodbye!")
        break
    if query:
        response = run_chain(query)
        response = response.replace("\n", " ").strip()[:300]
        print(f"\nAI Response: {response}")
        speak(response)
