import streamlit as st  # Import Streamlit for creating the web app
from PyPDF2 import PdfReader  # Import PdfReader to read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import RecursiveCharacterTextSplitter to split text into chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Import GoogleGenerativeAIEmbeddings for embedding generation
import google.generativeai as genai  # Import Google Generative AI library
from langchain_community.vectorstores import FAISS  # Import FAISS for creating and managing vector stores
from langchain_google_genai import ChatGoogleGenerativeAI  # Import ChatGoogleGenerativeAI for the conversational model
from langchain.chains.question_answering import load_qa_chain  # Import load_qa_chain to load a QA chain
from langchain.prompts import PromptTemplate  # Import PromptTemplate to create prompt templates
import os  # Import os for file and path management

# Set the configuration for the Streamlit page
st.set_page_config(page_title="My first RAG chatbot", layout="wide")

# Markdown text to display instructions and description on the Streamlit page
st.markdown("""
## My first RAG chatbot: Get instant insights from your Documents

This chatbot's code I got it from the [GitHub repository](https://github.com/SriLaxmi1993/Document-Genie-using-RAG-Framwork/tree/main) and I have made a few modifications in it. Follow the steps below to interact with the chatbot:
This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# Input field for the user to enter their Google API key, stored securely
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""  # Initialize an empty string to accumulate the text
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Create a PDF reader object for each uploaded PDF
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()  # Extract text from each page
            if extracted_text:
                text += extracted_text  # Append the extracted text to the accumulated text
    return text  # Return the accumulated text from all PDFs

def get_text_chunks(text):
    # Create a text splitter object to split the text into chunks of 10,000 characters with 1,000 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks  # Return the chunks

def get_vector_store(text_chunks, api_key):
    try:
        # Create embeddings for the text chunks using Google's Generative AI embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        # Create a FAISS vector store from the text chunks and their embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")  # Save the vector store locally
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")  # Display an error message if an exception occurs

def get_conversational_chain(api_key):
    # Define a prompt template for generating answers from the context and question
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context just say, "answer is not available in the context". Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    # Create a conversational AI model using Google's Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    # Create a prompt object using the prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load a question-answering chain using the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain  # Return the question-answering chain

def user_input(user_question, api_key):
    index_path = "faiss_index"  # Define the path to the FAISS index file
    if not os.path.exists(index_path):
        st.error("FAISS index file not found. Please upload and process the PDF documents first.")  # Display an error if the index file is not found
        return

    try:
        # Create embeddings for the FAISS index using Google's Generative AI embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        # Load the FAISS index from the local file
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)  # Perform a similarity search with the user's question
        chain = get_conversational_chain(api_key)  # Get the conversational chain
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)  # Generate a response using the chain
        st.write("Reply: ", response["output_text"])  # Display the response
    except Exception as e:
        st.error(f"An error occurred: {e}")  # Display an error message if an exception occurs

def main():
    st.header("PDF reader chatbot")  # Display the header for the main section

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")  # Input field for the user's question

    # Check if the user has entered a question and an API key
    if user_question and api_key:
        user_input(user_question, api_key)  # Process the user's question if both are provided
    elif user_question:
        st.error("Please enter your Google API Key.")  # Prompt the user to enter their API key if only a question is provided
    elif api_key:
        st.error("Please enter a question.")  # Prompt the user to enter a question if only the API key is provided

    with st.sidebar:
        st.title("Menu:")  # Display the title for the sidebar menu
        # File uploader for the user to upload PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        # Button for the user to submit and process the uploaded PDF files
        if st.button("Submit & Process", key="process_button"):
            if not api_key:
                st.error("Please enter your Google API Key.")  # Prompt the user to enter their API key if not provided
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")  # Prompt the user to upload at least one PDF file if not provided
            else:
                with st.spinner("Processing..."):  # Display a spinner while processing
                    try:
                        raw_text = get_pdf_text(pdf_docs)  # Extract text from the uploaded PDF files
                        text_chunks = get_text_chunks(raw_text)  # Split the extracted text into chunks
                        get_vector_store(text_chunks, api_key)  # Create a vector store from the text chunks
                        st.success("Processing completed successfully!")  # Display a success message upon completion
                    except Exception as e:
                        st.error(f"An error occurred while processing the PDFs: {e}")  # Display an error message if an exception occurs

if __name__ == "__main__":
    main()  # Call the main function to run the Streamlit app
