################# BASIC AI WITH WEB UI ####################
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() #Stores user-AI conversation history

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)
 # Function to run AI chat with memory
def run_chain(question):
    # Retrieve chat history manually
    chat_history_text ="\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    # Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit UI
st.title("AI Chatbot with Memory")
st.write("Ask me anything")

user_input = st.text_input("Your question")
if user_input:
    response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:**{response}")

# Show full chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")


################# BASIC AI WITH MEMORY ####################

# from langchain_community.chat_message_histories import ChatMessageHistory #Esto es para recordar los mensajes anteriores
# from langchain_core.prompts import PromptTemplate # Formatea la entrada
# from langchain_ollama import OllamaLLM

# # Load AI Model
# llm = OllamaLLM(model="mistral") #Change model to "llama3" or another if needed

# # Initialize Memory
# chat_history = ChatMessageHistory() # Stores user-AI conversation history

# #Define AI Chat Prompr
# prompt = PromptTemplate(
#     input_variables=["chat_history",  "question"],
#     template="Previous conversation: {chat_history}\nUser: {question}\nAI:" 
# )

# # Function to run AI chat with memory
# def run_chain(question):
#     # Retrieve chat history manually
#     chat_history_text ="\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

#     # Run the AI response generation
#     response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

#     # Stor new user input and AI response in memory
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response)

#     return response

# # Interactive CLI Chatbot
# print("\n AI Chatbot with Memory")
# print("Type 'exit' to stop.")

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() == "exit":
#         print("Arriverderci")
#         break
#     ai_response = run_chain(user_input)
#     print(f"\nAI: {ai_response}")


################# BASIC AI ####################

# # from langchain_ollama import OllamaLLM

# #  # Load AI model from Ollama
# # llm = OllamaLLM(model="mistral")

# # print("\n Welcome to your AI assistant!. Ask me anything. \n")

# # while True:
# #     question = input("\nYour question (or type 'exit' to stop); ")
# #     if question.lower() == 'exit':
# #         print("Arrivederci!")
# #         break
# #     response = llm.invoke(question)
# #     print("\n AI Response: ", response) 