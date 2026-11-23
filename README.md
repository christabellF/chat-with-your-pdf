# Chat with Your PDF – RAG-Based Question Answering System

## Project Overview
Chat with Your PDF is a Python-based web application that allows users to upload PDF documents and ask questions about their content. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded documents and generate context-based answers using a Large Language Model (LLM).

## Features
- Upload and interact with PDF documents.
- Ask questions based on uploaded PDF content.
- Retrieve relevant information using semantic similarity search.
- Generate answers using an LLM through the Groq API.
- Support multiple PDF documents.
- Maintain multi-turn conversations with chat history.
- Start new chats and manage previous conversations.

## Technologies Used
- **Python** – Application development
- **Streamlit** – Web application interface
- **PyMuPDF** – PDF text extraction
- **Sentence Transformers** – Text embeddings
- **NumPy** – Similarity calculations
- **Groq API** – LLM integration
- **python-dotenv** – Environment variable management

## How It Works
1. Users upload PDF documents.
2. The application extracts text from the PDFs.
3. Extracted text is divided into smaller chunks.
4. Sentence Transformers converts the chunks into numerical embeddings.
5. When a user asks a question, the application converts it into an embedding.
6. NumPy and cosine similarity identify the most relevant text chunks.
7. Retrieved information is passed to the LLM through the Groq API.
8. The LLM generates an answer based on the retrieved PDF content.

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/christabellF/chat-with-your-pdf.git
```

### 2. Navigate to the project folder
```bash
cd chat-with-your-pdf
```

### 3. Create a virtual environment
```bash
python -m venv .venv
```

### 4. Activate the virtual environment (Windows)
```bash
.venv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Configure the API key

Create a `.env` file in the project folder and add:

```text
GROQ_API_KEY=your_groq_api_key
```

### 7. Run the application
```bash
streamlit run app.py
```

## Security
The Groq API key is stored in an environment variable and is not included in the public repository.

