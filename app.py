import os
import json
import streamlit as st
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Page configuration
st.set_page_config(
    page_title="RAG Sales Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0b3c5d;
        text-align: center;
        padding: 1rem;
    }

    /* Chat Message Wrapper */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        line-height: 1.5;
        font-size: 1rem;
    }

    /* USER MESSAGE */
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1565c0;
        color: #0d47a1;
    }

    /* ASSISTANT MESSAGE */
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #2e7d32;
        color: #1b1b1b;
    }

    /* Intent Badge */
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: bold;
        margin-top: 0.5rem;
        border: 1px solid rgba(0,0,0,0.25);
    }

    /* SEARCH DB BADGE */
    .search-db { 
        background-color: #bbdefb; 
        color: #0b3c5d;
        border-color: #0b3c5d;
    }

    /* CUSTOMER HISTORY BADGE */
    .customer-history { 
        background-color: #c8e6c9; 
        color: #1b5e20;
        border-color: #1b5e20;
    }

    /* SUPPORT BADGE */
    .support { 
        background-color: #ffccbc; 
        color: #bf360c;
        border-color: #bf360c;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb {
        background: #b0bec5;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #78909c;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'intent_classifier' not in st.session_state:
    st.session_state.intent_classifier = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'token_counter' not in st.session_state:
    st.session_state.token_counter = None

# MongoDB connection with Atlas support
@st.cache_resource
def get_mongodb_connection():
    """Establish MongoDB connection (supports both local and Atlas)"""
    try:
        # Try to get MongoDB URI from secrets first (Streamlit Cloud)
        # then environment variables, then default to local
        MONGODB_URI = (
            st.secrets.get("MONGODB_URI") if hasattr(st, 'secrets') and "MONGODB_URI" in st.secrets
            else os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        )
        
        DB_NAME = (
            st.secrets.get("DB_NAME") if hasattr(st, 'secrets') and "DB_NAME" in st.secrets
            else os.getenv("DB_NAME", "rag_chatbot_db")
        )
        
        # Connect with a longer timeout for Atlas
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True
        )
        
        # Test connection
        client.admin.command('ping')
        st.success(f"✅ Connected to MongoDB: {DB_NAME}")
        
        return client[DB_NAME]
    except ConnectionFailure as e:
        st.error(f"❌ Failed to connect to MongoDB: {e}")
        st.info("💡 Please check your MongoDB Atlas connection string in Streamlit secrets")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error connecting to MongoDB: {e}")
        return None

def upload_json_to_mongodb(json_file, db) -> int:
    """Upload JSON file data to MongoDB"""
    try:
        data = json.loads(json_file.read())
        
        if isinstance(data, list):
            documents = data
        else:
            documents = [data]
        
        # Limit to 100 records for demo
        documents = documents[:100]
        
        transactions_collection = db["transactions"]
        customers_collection = db["customers"]
        products_collection = db["products"]
        
        # Clear existing data
        transactions_collection.delete_many({})
        customers_collection.delete_many({})
        products_collection.delete_many({})
        
        customers_dict = {}
        products_dict = {}
        transactions = []
        
        for doc in documents:
            customer_id = doc.get("Customer ID", "UNKNOWN")
            customer_name = doc.get("Customer name", "Unknown")
            
            if customer_id not in customers_dict:
                customers_dict[customer_id] = {
                    "customer_id": customer_id,
                    "name": customer_name,
                    "email": doc.get("Email", "N/A"),
                    "phone": doc.get("Phone", "N/A"),
                    "city": doc.get("City", "N/A"),
                    "loyalty_tier": doc.get("Loyalty_Tier", "Regular"),
                    "created_at": datetime.now()
                }
            
            product_id = doc.get("ID_product", "UNKNOWN")
            if product_id not in products_dict:
                products_dict[product_id] = {
                    "product_id": product_id,
                    "name": doc.get("Product", "Unknown"),
                    "category": doc.get("Category", "N/A"),
                    "sku": doc.get("SKUs", "N/A"),
                    "cogs": doc.get("COGS", 0),
                    "margin_percent": doc.get("Margin_per_piece_percent", 0),
                    "created_at": datetime.now()
                }
            
            transaction = {
                "invoice_number": doc.get("Invoice Number", "N/A"),
                "txn_number": doc.get("Txn_No", "N/A"),
                "customer_id": customer_id,
                "customer_name": customer_name,
                "product_id": product_id,
                "product_name": doc.get("Product", "Unknown"),
                "category": doc.get("Category", "N/A"),
                "quantity": doc.get("Quantity_piece", 0),
                "gross_amount": doc.get("Gross_Amount", 0),
                "discount_percentage": doc.get("Discount_Percentage", 0),
                "total_amount": doc.get("Total Amount", 0),
                "gst": doc.get("GST", 0),
                "payment_mode": doc.get("Payment_mode", "N/A"),
                "date_of_purchase": doc.get("Date_of_purchase", datetime.now().isoformat()),
                "channel": doc.get("Channel", "N/A"),
                "store_location": doc.get("Store_location", "N/A"),
                "mode": doc.get("Mode", "N/A"),
                "status": "completed",
                "created_at": datetime.now()
            }
            transactions.append(transaction)
        
        if customers_dict:
            customers_collection.insert_many(list(customers_dict.values()))
        if products_dict:
            products_collection.insert_many(list(products_dict.values()))
        if transactions:
            result = transactions_collection.insert_many(transactions)
            return len(result.inserted_ids)
        
        return 0
    except Exception as e:
        st.error(f"Error uploading data: {e}")
        return 0

def mongodb_to_searchable_text(db) -> List[str]:
    """Convert MongoDB transactions to searchable text chunks"""
    transactions_collection = db["transactions"]
    transactions = list(transactions_collection.find())
    
    if not transactions:
        raise ValueError("No transactions found in MongoDB")
    
    texts = []
    for txn in transactions:
        text = f"""
Transaction Details:
- Invoice: {txn.get('invoice_number')}
- Customer: {txn.get('customer_name')} (ID: {txn.get('customer_id')})
- Product: {txn.get('product_name')} (Category: {txn.get('category')})
- Quantity: {txn.get('quantity')} pieces
- Total Amount: ${txn.get('total_amount'):.2f}
- GST: ${txn.get('gst'):.2f}
- Gross Amount: ${txn.get('gross_amount'):.2f}
- Discount: {txn.get('discount_percentage'):.2f}%
- Payment Mode: {txn.get('payment_mode')}
- Store Location: {txn.get('store_location')}
- Channel: {txn.get('channel')}
- Purchase Date: {txn.get('date_of_purchase')}
- Status: {txn.get('status')}
"""
        texts.append(text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text("\n".join(texts))
    
    return chunks

class EmbeddingIntentClassifier:
    """Intent classifier using embedding similarity"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        
        self.intent_templates = {
            "SEARCH_DB": [
                "What products do you have?",
                "Show me sales data",
                "How many items sold?",
                "What is the price of product?",
                "List all products in category",
                "Show inventory",
                "What are the top selling products?",
                "Total sales amount",
                "How much revenue?",
                "Product information"
            ],
            "CUSTOMER_HISTORY": [
                "Show my purchase history",
                "What did I buy?",
                "My previous orders",
                "My transaction history",
                "Orders for customer John Doe",
                "Show transactions for customer ID",
                "My account purchases",
                "What have I ordered before?",
                "My past invoices"
            ],
            "SUPPORT": [
                "I have a problem",
                "Need help with my order",
                "Product is broken",
                "Issue with delivery",
                "Customer service needed",
                "Contact support team",
                "File a complaint",
                "Not working properly"
            ]
        }
        
        self.intent_embeddings = {}
        for intent, templates in self.intent_templates.items():
            embeddings = self.embeddings_model.embed_documents(templates)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
    
    def classify(self, question: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify intent using cosine similarity"""
        question_embedding = self.embeddings_model.embed_query(question)
        
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            similarity = cosine_similarity(
                [question_embedding], 
                [intent_embedding]
            )[0][0]
            similarities[intent] = similarity
        
        best_intent = max(similarities, key=similarities.get)
        confidence = similarities[best_intent]
        
        return best_intent, confidence, similarities

class TokenCounter:
    """Simple token counter utility"""
    
    def __init__(self):
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(text) // 4

@st.cache_resource
def build_rag_model(_db):
    """Build RAG model with Gemini API and FAISS"""
    # Try to get API key from secrets first, then environment variables
    GOOGLE_API_KEY = (
        st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets
        else os.getenv("GOOGLE_API_KEY")
    )
    
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY not found in secrets or environment variables")
        st.info("💡 Please add GOOGLE_API_KEY to your Streamlit secrets")
        return None, None, None
    
    try:
        chunks = mongodb_to_searchable_text(_db)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
            max_output_tokens=1500
        )
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful customer service assistant for an e-commerce platform. "
                "Based on the following transaction data from our database, "
                "provide a clear and helpful answer to the customer's question. "
                "Keep your answer concise and relevant to sales and transactions.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        intent_classifier = EmbeddingIntentClassifier(embeddings)
        
        return qa_chain, llm, intent_classifier
    except Exception as e:
        st.error(f"Error building RAG model: {e}")
        return None, None, None

def handle_search_db(question: str, qa_chain, chat_history: List[Tuple[str, str]]) -> str:
    """Search database and answer questions"""
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    answer = result.get("answer") or result.get("result") or "⚠️ No relevant information found"
    
    with st.expander("📎 View Source Documents"):
        source_docs = result.get("source_documents", [])
        if source_docs:
            for i, doc in enumerate(source_docs, 1):
                st.text_area(f"Source {i}", doc.page_content, height=200)
        else:
            st.info("No source documents found")
    
    return answer

def handle_customer_history(question: str, llm, db) -> str:
    """Retrieve customer purchase history"""
    extract_prompt = f"""From this question, extract:
1. Customer identifier (name, ID, or email) - if present
2. Invoice number - if present

Question: "{question}"

Respond in this format:
CUSTOMER: [customer name/ID/email or NOT_FOUND]
INVOICE: [invoice number or NOT_FOUND]"""
    
    response = llm.invoke(extract_prompt)
    extraction_result = response.content.strip()
    
    lines = extraction_result.split('\n')
    customer_identifier = "NOT_FOUND"
    invoice_number = "NOT_FOUND"
    
    for line in lines:
        if line.startswith("CUSTOMER:"):
            customer_identifier = line.replace("CUSTOMER:", "").strip()
        elif line.startswith("INVOICE:"):
            invoice_number = line.replace("INVOICE:", "").strip()
    
    transactions_collection = db["transactions"]
    customers_collection = db["customers"]
    
    # Handle invoice search
    if invoice_number != "NOT_FOUND":
        txn = transactions_collection.find_one({"invoice_number": invoice_number})
        
        if not txn:
            return f"❌ Invoice '{invoice_number}' not found in database."
        
        customer_id = txn.get("customer_id")
        customer = customers_collection.find_one({"customer_id": customer_id})
        
        if not customer:
            customer = {}
        
        history = f"""
{'='*70}
INVOICE DETAILS
{'='*70}
Invoice Number: {txn.get('invoice_number')}
Transaction ID: {txn.get('txn_number')}
Date: {txn.get('date_of_purchase')}
Status: {txn.get('status')}

CUSTOMER INFORMATION:
Name: {customer.get('name', 'Unknown')}
Customer ID: {customer.get('customer_id', 'N/A')}
Email: {customer.get('email', 'N/A')}
Phone: {customer.get('phone', 'N/A')}

ORDER DETAILS:
Product: {txn.get('product_name')}
Category: {txn.get('category')}
Quantity: {txn.get('quantity')} pieces
Total Amount: ${txn.get('total_amount'):.2f}
Payment Mode: {txn.get('payment_mode')}
"""
        return history
    
    # Handle customer search
    if customer_identifier == "NOT_FOUND":
        return "⚠️ Please provide a customer name, ID, or invoice number"
    
    customer = customers_collection.find_one({
        "$or": [
            {"name": {"$regex": customer_identifier, "$options": "i"}},
            {"customer_id": customer_identifier},
            {"email": {"$regex": customer_identifier, "$options": "i"}}
        ]
    })
    
    if not customer:
        return f"❌ Customer '{customer_identifier}' not found in database."
    
    customer_id = customer.get("customer_id")
    customer_name = customer.get("name", "Unknown")
    
    txn_list = list(transactions_collection.find(
        {"customer_id": customer_id}
    ).sort("date_of_purchase", -1))
    
    if not txn_list:
        return f"""
✓ Customer Found: {customer_name}
Email: {customer.get('email', 'N/A')}
Phone: {customer.get('phone', 'N/A')}

❌ No purchase history found for this customer."""
    
    history = f"""
{'='*70}
CUSTOMER PURCHASE HISTORY
{'='*70}
Name: {customer_name}
Email: {customer.get('email', 'N/A')}
Phone: {customer.get('phone', 'N/A')}
City: {customer.get('city', 'N/A')}
Loyalty Tier: {customer.get('loyalty_tier', 'N/A')}

TRANSACTIONS ({len(txn_list)} total):
{'='*70}
"""
    
    total_spent = 0
    for idx, txn in enumerate(txn_list, 1):
        history += f"""
{idx}. Invoice: {txn.get('invoice_number')}
   Product: {txn.get('product_name')} ({txn.get('category')})
   Quantity: {txn.get('quantity')} pieces
   Amount: ${txn.get('total_amount'):.2f}
   Date: {txn.get('date_of_purchase')}
"""
        total_spent += txn.get('total_amount', 0)
    
    history += f"""
{'='*70}
Total Spent: ${total_spent:.2f}
{'='*70}
"""
    
    return history

def handle_support_request(db) -> str:
    """Handle customer support issues"""
    support_tickets_collection = db["support_tickets"]
    
    with st.form("support_form"):
        st.subheader("📞 Customer Support Request")
        issue = st.text_area("Describe your issue:", height=100)
        name = st.text_input("Your name:")
        email = st.text_input("Your email:")
        phone = st.text_input("Your phone (optional):")
        
        submitted = st.form_submit_button("Submit Ticket")
        
        if submitted:
            if not issue or not name or not email:
                return "⚠️ Please fill in all required fields"
            
            ticket_num = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            priority = "high" if any(w in issue.lower() for w in ["urgent", "critical", "emergency", "broken"]) else "normal"
            
            ticket_doc = {
                "ticket_number": ticket_num,
                "customer_name": name,
                "customer_email": email,
                "customer_phone": phone,
                "issue": issue,
                "status": "open",
                "priority": priority,
                "created_at": datetime.now()
            }
            
            support_tickets_collection.insert_one(ticket_doc)
            
            return f"""
✓ SUPPORT TICKET CREATED
Ticket Number: {ticket_num}
Priority: {priority.upper()}
Status: OPEN

Your support ticket has been logged.
Expected Response Time: 1-2 business hours

📞 Phone: 1-800-SUPPORT
📧 Email: support@company.com
"""
    
    return ""

# Main Application
def main():
    st.markdown('<h1 class="main-header">🤖 RAG Sales & Support Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database status
        if st.session_state.db is None:
            st.session_state.db = get_mongodb_connection()
        
        if st.session_state.db is not None:
            st.success("✅ MongoDB Connected")
        else:
            st.error("❌ MongoDB Disconnected")
            st.stop()
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
        
        if uploaded_file and not st.session_state.data_loaded:
            with st.spinner("Uploading data to MongoDB..."):
                count = upload_json_to_mongodb(uploaded_file, st.session_state.db)
                if count > 0:
                    st.success(f"✅ Loaded {count} transactions")
                    st.session_state.data_loaded = True
                    st.rerun()
        
        # Build RAG model
        if st.session_state.data_loaded and st.session_state.qa_chain is None:
            with st.spinner("Building RAG model..."):
                qa_chain, llm, intent_classifier = build_rag_model(st.session_state.db)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.llm = llm
                    st.session_state.intent_classifier = intent_classifier
                    st.session_state.token_counter = TokenCounter()
                    st.success("✅ RAG Model Ready")
        
        # Stats
        if st.session_state.data_loaded:
            st.subheader("📊 Statistics")
            transactions_collection = st.session_state.db["transactions"]
            customers_collection = st.session_state.db["customers"]
            
            txn_count = transactions_collection.count_documents({})
            cust_count = customers_collection.count_documents({})
            
            col1, col2 = st.columns(2)
            col1.metric("Transactions", txn_count)
            col2.metric("Customers", cust_count)
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.data_loaded:
        st.info("👆 Please upload a JSON file to get started")
        st.markdown("""
        ### ✨ I can help you with:
        - 🔍 Answer questions about products, sales, and transactions
        - 📊 Show complete purchase history for customers
        - 🆘 Connect with customer support
        """)
        return
    
    if st.session_state.qa_chain is None:
        st.warning("⏳ Building RAG model, please wait...")
        return
    
    # Display chat history
    for idx, (role, message, intent, confidence, similarities) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(
                f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{message}</div>', 
                unsafe_allow_html=True
            )
            
            if intent and confidence and similarities:
                intent_class = intent.lower().replace("_", "-")
                st.markdown(
                    f'<span class="intent-badge {intent_class}">Intent: {intent} ({confidence:.2%})</span>', 
                    unsafe_allow_html=True
                )
                
        else:
            st.markdown(
                f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong><br><pre>{message}</pre></div>', 
                unsafe_allow_html=True
            )
    
    # Chat input
    question = st.chat_input("Ask me anything about sales, products, or customer history...")
    
    if question:
        # Classify intent
        intent, confidence, similarities = st.session_state.intent_classifier.classify(question)
        
        # Add user message to history
        st.session_state.chat_history.append(("user", question, intent, confidence, similarities))
        
        # Process based on intent
        with st.spinner("Processing your request..."):
            if intent == "SEARCH_DB":
                chat_hist = []
                answer = handle_search_db(question, st.session_state.qa_chain, chat_hist)
                
            elif intent == "CUSTOMER_HISTORY":
                answer = handle_customer_history(question, st.session_state.llm, st.session_state.db)
                
            elif intent == "SUPPORT":
                answer = handle_support_request(st.session_state.db)
            else:
                answer = "I'm not sure which intent this falls under."
            
            # Add assistant response to history
            st.session_state.chat_history.append(("assistant", answer, None, None, None))
        
        st.rerun()

if __name__ == "__main__":
    main()