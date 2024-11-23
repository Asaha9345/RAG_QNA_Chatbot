import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import tempfile

# Streamlit UI: Title for the application
st.title("RAG Application")

# Function to clean text by removing unwanted characters and extra spaces
def clean_text(text):
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'^[@#$%*?+-]',' ',text)
    return text.strip()

# File uploader widget for accepting PDF documents
st.sidebar.title("Key Information")
st.sidebar.text("Please enter the information\nto start the application.")
huggingface_id=st.sidebar.text_input("Enter your Huggingface API Key",type='password')
groq_id=st.sidebar.text_input("Enter your Groq API Key",type='password')
if huggingface_id and groq_id:
    try:
        document=st.file_uploader("Please choose a pdf file to start your Q&A",type='pdf')
        if document is not None:
            st.write("Your document is being prepared..")
            # Temporary file creation for handling uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(document.read())
                temp_file_path = tmp_file.name
    
            try:
                # Define the prompt for the ChatGroq language model
                prompt=ChatPromptTemplate.from_template(
                    """ You are a helpfull AI assistant. try to answer the user questions.
                    if you do not know the answer then just say that please ask relevent question.
                    If anything is asked outside the context of the document, You should inform that to ask releb=vent question.
                    <context>
                    {context}
                    <context>
                    Question:{input}"""
                    )
                # Load environment variables for Hugging Face
                os.environ['HF_TOKEN']=str(huggingface_id)
                # Load the PDF document into LangChain's pipeline
                loader=PyPDFLoader(temp_file_path)
                docs=loader.load()
                # Split the document into smaller chunks for processing
                splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
                # Initialize the ChatGroq language model
                llm=ChatGroq(model='Gemma2-9b-it',groq_api_key=str(groq_id))
                # Create embeddings for the document using Hugging Face's model
                embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-l6-v2')
                # Clean the content of each document page
                for doc in docs:
                    doc.page_content=clean_text(doc.page_content)
                # Split the cleaned document into smaller chunks
                final_doc=splitter.split_documents(docs)
                # Create a FAISS vector store from the document chunks
                vector_store=FAISS.from_documents(final_doc,embedding)
                # Define the retriever for fetching relevant document chunks
                retriever=vector_store.as_retriever()
                # Create a document processing chain with the LLM and custom prompt
                document_chain=create_stuff_documents_chain(llm,prompt)
                # Build a Retrieval-Augmented Generation (RAG) pipeline
                rag_chain=create_retrieval_chain(retriever,document_chain)
                # Input widget for user queries
                user=st.text_input("What's your question?")
                if user:
                    # Pass the user input through the RAG chain to get an answer
                    answer=rag_chain.invoke({'input':user})
                    if 'answer' in answer:
                        st.write(answer['answer'])
                    else:
                        st.write('No answer could be generated')
            finally:
                # Ensure the temporary file is deleted after processing is completed
                os.remove(temp_file_path)
    except:
        st.error("Please enter the API key correctly")
