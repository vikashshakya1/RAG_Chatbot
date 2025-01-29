# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Flask, Faiss, and MySQL. This chatbot retrieves relevant text chunks from a pre-indexed database and generates responses based on the retrieved information. It supports conversational history storage using MySQL.

## Features
- **Retrieval-Augmented Generation (RAG):** Combines retrieval and generative models for accurate responses.
- **Efficient Similarity Search:** Uses FAISS for fast nearest-neighbor lookup.
- **Powerful Embeddings:** Utilizes Sentence-BERT for vector embeddings.
- **Database Storage:** Stores chat history in MySQL for persistent conversations.
- **Flask API:** Provides easy-to-use endpoints for chat interaction and history retrieval.
- **Scalable and Extendable:** Can integrate more advanced models and external knowledge bases.

---

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a Virtual Environment & Install Dependencies
```sh
python -m venv rag_env
source rag_env/bin/activate   # On macOS/Linux
rag_env\Scripts\activate     # On Windows
pip install -r requirements.txt
```

---

## Database Setup (MySQL)

### 3. Install MySQL
Ensure MySQL is installed and running. You can download it from [MySQL Downloads](https://dev.mysql.com/downloads/).

### 4. Create the Database & Table
Run the following SQL commands to set up the database and tables:
```sql
CREATE DATABASE rag_chatbot;
USE rag_chatbot;

CREATE TABLE chat_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role ENUM('user', 'system'),
    content TEXT
);
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root and add:
```ini
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=12345
MYSQL_DATABASE=rag_chatbot
```
Ensure the credentials match your MySQL setup.

---

## Running the Chatbot Locally

### 6. Start the Flask Server
```sh
python rag_chatbot.py
```
The server will start running at `http://127.0.0.1:5000`.

---

## API Usage

### 7. Test the `/chat` Endpoint
Send a POST request with a JSON body containing a query:
```json
{
    "query": "What is climate change?"
}
```
Example using `curl`:
```sh
curl -X POST "http://127.0.0.1:5000/chat" -H "Content-Type: application/json" -d '{"query": "What is climate change?"}'
```
Response format:
```json
{
    "response": "Climate change refers to long-term shifts in temperature and weather patterns..."
}
```

### 8. Retrieve Chat History
Send a GET request to fetch stored conversation history:
```sh
curl -X GET "http://127.0.0.1:5000/history"
```
Response format:
```json
[
    { "timestamp": "2025-01-29T22:49:31", "role": "user", "content": "What is climate change?" },
    { "timestamp": "2025-01-29T22:49:32", "role": "system", "content": "Climate change refers to..." }
]
```

---

## Notes
- Ensure that the `vector_store.index` and `metadata.pkl` files are correctly loaded before running the chatbot.
- The chatbot can be extended with more powerful models like GPT-4 or fine-tuned LLMs.
- Modify `rag_chatbot.py` to adjust retrieval parameters or integrate additional data sources.

---

## License
This project is open-source and available under the MIT License.

## Contributors
- **Vikash Shakya** - Developer
For any issues, create a GitHub issue or contact the maintainer.

Happy coding! 🚀

