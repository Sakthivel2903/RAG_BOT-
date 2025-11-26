# 🤖 RAG Sales & Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, Google Gemini, and MongoDB Atlas for sales and customer support.

## Features

- 🔍 **Intelligent Search**: Query products, sales data, and transactions
- 📊 **Customer History**: View complete purchase history and invoices
- 🆘 **Support Tickets**: Create and manage support requests
- 🎯 **Intent Classification**: Automatic detection of user intent using embeddings
- 💾 **MongoDB Atlas**: Cloud database for scalable storage

## Prerequisites

- Python 3.9+
- Google Gemini API key
- MongoDB Atlas account

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd rag-sales-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure MongoDB Atlas

1. Create a free MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster (M0 FREE tier)
3. Create a database user with read/write permissions
4. Whitelist your IP address (or use 0.0.0.0/0 for Streamlit Cloud)
5. Get your connection string

### 4. Set Up Environment Variables

Create a `.streamlit/secrets.toml` file:

```toml
MONGODB_URI = "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "rag_chatbot_db"
GOOGLE_API_KEY = "your-google-gemini-api-key"
```

### 5. Run Locally

```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch (main), and main file (app.py)
5. Click "Advanced settings"
6. Add secrets:
   ```toml
   MONGODB_URI = "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
   DB_NAME = "rag_chatbot_db"
   GOOGLE_API_KEY = "your-google-gemini-api-key"
   ```
7. Click "Deploy!"

## Usage

1. **Upload Data**: Upload a JSON file containing transaction data
2. **Ask Questions**: 
   - "What products do you have?"
   - "Show purchase history for John Doe"
   - "I need help with my order"
3. **View Results**: Get AI-powered responses with source citations

## Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml      # Configuration secrets (not in repo)
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **Google Gemini**: LLM and embeddings
- **FAISS**: Vector similarity search
- **MongoDB Atlas**: Cloud database
- **scikit-learn**: Cosine similarity for intent classification

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first.
