import streamlit as st  # Streamlit library for creating the web app interface
from translate import Translator  # Translator for multi-language support
import speech_recognition as sr  # Speech recognition for handling audio input
from langchain.document_loaders import PyPDFLoader, DirectoryLoader  # Loading PDF files or directories
from langchain import PromptTemplate  # Allows setting up a custom prompt template for the chatbot
from langchain.embeddings import HuggingFaceEmbeddings  # Embedding model for vectorizing text inputs
from langchain.vectorstores import FAISS  # Vector store for efficient similarity search and retrieval
from langchain.llms import CTransformers  # CTransformers for loading language models
from langchain.chains import RetrievalQA  # Enables retrieval-based question-answering
import os  # OS module for handling file operations

# Path to FAISS vector store (adjust based on your storage)
DB_FAISS_PATH = 'vectorstore/db_faiss'  

# Template that guides the bot's responses with a set of instructions
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that I don't know, sorry for the inconvenience, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Dictionary to store responses along with associated download links
response_download_links = {}

def set_custom_prompt():
    """
    Initializes a prompt template with custom instructions.
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt  # Returns the created prompt template

def retrieval_qa_chain(llm, prompt, db):
    """
    Sets up a question-answering chain that uses retrieval to find answers.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Language model
        chain_type='stuff',  # Type of chain used for retrieval
        retriever=db.as_retriever(search_kwargs={'k': 2}),  # Retrieve top 2 similar documents
        return_source_documents=True,  # Include source documents for additional context
        chain_type_kwargs={'prompt': prompt}  # Use custom prompt for responses
    )
    return qa_chain  # Returns the configured QA chain object

def load_llm():
    """
    Loads a language model with CTransformers, setting parameters for the model.
    """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",  # Specify the Llama model
        model_type="llama",  # Define model type
        max_new_tokens=10002,  # Maximum tokens in response
        temperature=1.0  # Controls randomness in generation
    )
    return llm  # Returns the loaded language model

def qa_bot():
    """
    Initializes and returns the chatbot with embeddings and language model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Run on CPU
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)  # Load FAISS vector store
    llm = load_llm()  # Load the language model
    qa_prompt = set_custom_prompt()  # Set up custom prompt
    qa = retrieval_qa_chain(llm, qa_prompt, db)  # Create the QA chain
    return qa  # Returns the complete QA bot

def final_result(query):
    """
    Generates a response from the chatbot based on the query.
    """
    qa_result = qa_bot()  # Initialize the chatbot
    response = qa_result({'query': query})  # Get response for the query
    result = response.get('result')  # Extract the main answer
    source = response.get('source_documents')  # Extract the source documents
    return result, source  # Return result and source

def translate_text(text, lang):
    """
    Translates text into the selected language, handling character limits.
    """
    max_char_limit = 500  # Set maximum characters per translation segment
    translated_segments = []  # List for storing translated segments

    # Split text into manageable segments
    segments = [text[i:i + max_char_limit] for i in range(0, len(text), max_char_limit)]
    
    # Translate each segment individually
    for segment in segments:
        translator = Translator(to_lang=lang)  # Initialize Translator for target language
        translated_segment = translator.translate(segment)  # Translate the segment
        translated_segments.append(translated_segment)  # Store the translated segment
    
    translated_text = ' '.join(translated_segments)  # Combine translated segments
    return translated_text  # Return the final translated text

# Dictionary of language codes for multi-language support
language_codes = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Odia (Oriya)': 'or',
    'Malayalam': 'ml',
    'Punjabi': 'pa'
}

# Streamlit UI setup
st.title("💬 Mining Bot")  # Set the title for the app
st.caption("🚀 A Chatbot powered by LLMA2")  # Subtitle for additional context

# Sidebar for language selection
languages = list(language_codes.keys())  # Get list of supported languages
selected_language = st.sidebar.selectbox('Select a language:', languages)  # Language selection dropdown

# Initialize session state to store chat messages if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display each message in the chat history
for msg in st.session_state.messages:
    st.write(f"{msg['role']}: {msg['content']}")  # Display stored messages

# Placeholder for the "Speak" button at the bottom
speak_placeholder = st.empty()  # Placeholder for positioning "Speak" button
record_audio = speak_placeholder.button("Speak")  # Button to start audio recording

# Audio input handling
if record_audio:  # If the "Speak" button is clicked
    r = sr.Recognizer()  # Initialize recognizer
    with sr.Microphone() as source:  # Use the default microphone
        st.write("Listening...")  # Indicate listening state
        audio = r.listen(source)  # Capture audio input
        st.write("Audio recording complete.")  # Indicate completion of audio capture

    try:
        user_input = r.recognize_google(audio)  # Convert audio to text using Google
        st.session_state.messages.append({"role": "user", "content": user_input})  # Add user input to session state
        st.write(f"user: {user_input}")  # Display user input

        # Process the query and retrieve response and sources
        response, sources = final_result(user_input)

        # Get language code for selected language and translate response
        selected_language_code = language_codes[selected_language]
        translated_response = translate_text(response, selected_language_code)

        # Display assistant's response and add it to session state
        msg = {"role": "assistant", "content": translated_response}
        st.session_state.messages.append(msg)
        st.write(f"assistant: {msg['content']}")  # Display assistant's translated response

    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")  # Handle unrecognized audio
    except sr.RequestError:
        st.write("Sorry, I encountered an error while processing the audio.")  # Handle API errors

# Text input handling for queries
if prompt := st.text_input("User Input"):  # Text input for user queries
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add user text input to session
    st.write(f"user: {prompt}")  # Display user text input

    # Process the query and get response and sources
    response, sources = final_result(prompt)
    selected_language_code = language_codes[selected_language]  # Get selected language code

    # Translate the response
    translated_response = translate_text(response, selected_language_code)
    st.write(f"assistant: {translated_response}")  # Display translated response
    st.session_state.messages.append({"role": "assistant", "content": translated_response})  # Store response

# Render stored responses with download buttons
for response in response_download_links:
    st.download_button("Download Response", response_download_links[response], key=response)
