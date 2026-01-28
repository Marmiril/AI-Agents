import requests # For http request
from bs4 import BeautifulSoup # HTML analizes an extract text from websites
import streamlit as st # Python library for interactive web applications building
from langchain_ollama import OllamaLLM

# Load AI model
llm = OllamaLLM(model="mistral") # Or whatever model if needed

# Function to scrape a website
def scrape_website(url):
    try:
        st.write(f"Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return f"Failed to fetch {url}"
        
        # Extract text content
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000] # Limit characters to avoid overloading AI
    except Exception as e:
        return f"Error: {str(e)}"
    
# Function so summarize content using AI
def summarize_content(content):
    st.write("Summarizing content...")
    return llm.invoke(f"Summarize the following content: \n\n{content[:1000]}")

# StreamLit Web UI
st.title("AI-Powered Web Scraper")
st.write("Enter a website URL below and get a summarized version!")

# User input
url = st.text_input("Enter Website URL")
if url:
    content = scrape_website(url)

    if "Failed" in content or "Error" in content:
        st.write(content)
    else: 
        summary = summarize_content(content)
        st.subheader("Website summary")
        st.write(summary)