# RAG Application (Retrieval-Augmented Generation)

This application is a **Retrieval-Augmented Generation (RAG)** system built using **Streamlit**, **LangChain**, and integrations with Hugging Face and Groq APIs. It enables users to upload PDF documents and ask questions about the content. The app retrieves relevant sections from the document and generates contextual answers using AI.

---

## Features
- **Document Upload**: Upload PDF files to analyze their content.
- **Question Answering**: Ask questions based on the uploaded document.
- **Language Model Integration**:
  - Uses **ChatGroq** for AI-generated responses.
  - Leverages **Hugging Face Embeddings** for text vectorization.
- **Dynamic API Key Input**: Securely input API keys for Hugging Face and Groq.
- **Interactive UI**: Built with **Streamlit** for a simple and user-friendly experience.

---

## Prerequisites
- **Python 3.8 or above** is required.
- Active API keys for:
  - Hugging Face (HuggingFaceEmbeddings)
  - Groq (ChatGroq)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Asaha9345/RAG_QNA_Chatbot.git
cd rag-application
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory and add the following:
```plaintext
HF_TOKEN=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```
Alternatively, you can enter the API keys directly in the app's sidebar when running the application.

---

## Usage

### 1. Start the Streamlit Application
Run the app using:
```bash
streamlit run app.py
```

### 2. Interact with the App
- Upload a **PDF document**.
- Enter your **Hugging Face** and **Groq API keys** in the sidebar.
- Ask questions about the content of the document through the text input box.

---

## Project Structure
```plaintext
.
├── app.py               # Main application code
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Excluded files for version control
```

---

## Known Issues
- Ensure the API keys are valid; otherwise, a 401 Authentication Error may occur.
- Large PDF files might slow down processing due to document splitting and vectorization.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## Acknowledgments
- **LangChain** for providing tools to build document pipelines.
- **Streamlit** for the interactive web interface.
- **Hugging Face** and **Groq** for AI model support.

---

Feel free to contact [asaha9345@gmail.com] for questions or suggestions!
