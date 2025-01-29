import os
import faiss
import pickle
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import mysql.connector
from mysql.connector import pooling
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables or use defaults
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "12345"),
    "database": os.getenv("MYSQL_DATABASE", "rag_chatbot"),
}

# MySQL connection pooling
db_pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **MYSQL_CONFIG)

def get_db_connection():
    """Get a database connection from the pool."""
    try:
        return db_pool.get_connection()
    except mysql.connector.Error as e:
        logging.error(f"❌ MySQL connection error: {e}")
        return None

# Load embedding model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("✅ Sentence transformer model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Load Faiss index and metadata
text_chunks = []
index = None
try:
    index = faiss.read_index("vector_store.index")
    with open("metadata.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    logging.info("✅ Faiss index and metadata loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading Faiss index or metadata: {e}")

# Initialize Flask app
app = Flask(__name__)

def setup_database():
    """Ensure chat history table exists."""
    db = get_db_connection()
    if db:
        try:
            with db.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        role ENUM('user', 'system'),
                        content TEXT
                    );
                """)
                db.commit()
            logging.info("✅ Database setup complete.")
        except mysql.connector.Error as e:
            logging.error(f"❌ MySQL setup error: {e}")
        finally:
            db.close()

setup_database()

def retrieve_relevant_chunks(query, top_k=3):
    """Finds the most relevant text chunks for a query."""
    if index is None or not text_chunks:
        logging.warning("⚠️ Knowledge base not loaded or empty.")
        return ["No knowledge base available."]
    
    try:
        query_embedding = model.encode([query])
        _, indices = index.search(np.array(query_embedding), top_k)
        return [text_chunks[i] for i in indices[0]]
    except Exception as e:
        logging.error(f"❌ Retrieval error: {e}")
        return ["Error retrieving relevant information."]

@app.route('/')
def home():
    """Default route to serve frontend (index.html)."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests and retrieves relevant text chunks."""
    data = request.json
    user_query = data.get('query', '').strip()

    if not user_query:
        logging.error("❌ Query is empty!")
        return jsonify({"error": "Query cannot be empty"}), 400

    logging.info(f"User query: {user_query}")
    retrieved_texts = retrieve_relevant_chunks(user_query)

    if not retrieved_texts or "No knowledge base available." in retrieved_texts:
        logging.warning("⚠️ No relevant information retrieved.")
    
    generated_response = " ".join(retrieved_texts) if retrieved_texts else "No relevant information found."
    logging.info(f"Generated response: {generated_response}")

    # Store conversation in MySQL
    db = get_db_connection()
    if db:
        try:
            with db.cursor() as cursor:
                cursor.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s)", ("user", user_query))
                cursor.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s)", ("system", generated_response))
                db.commit()
            logging.info("✅ User query and response saved to database.")
        except mysql.connector.Error as e:
            logging.error(f"❌ Database insertion error: {e}")
        finally:
            db.close()

    return jsonify({"response": generated_response, "retrieved_chunks": retrieved_texts})


@app.route('/history', methods=['GET'])
def history():
    """Fetch chat history from MySQL."""
    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT role, content, timestamp FROM chat_history ORDER BY timestamp DESC")
            history = cursor.fetchall()
            if not history:
                return jsonify({"error": "No chat history available."}), 404
        return jsonify({"history": [{"role": r, "content": c, "timestamp": str(t)} for r, c, t in history]})
    except mysql.connector.Error as e:
        logging.error(f"❌ Database retrieval error: {e}")
        return jsonify({"error": "Could not retrieve chat history"}), 500
    finally:
        db.close()

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent errors."""
    return '', 204  # No Content response

if __name__ == "__main__":
    app.run(debug=True)
