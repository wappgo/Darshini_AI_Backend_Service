A smart AI assistant for the Kumbh Mela festival, built with FastAPI and LangChain.

## Tech Stack
- **Backend:** FastAPI
- **LLM:** LangChain & Groq
- **Databases:** PostgreSQL (for items) & MongoDB (for chat history)
- **Auth:** JWT

## Quickstart

### 1. Clone & Install Dependencies
```bash
git clone <repository-url>
cd darshini-ai
pip install -r requirements.txt

2. Configure Environment
Create a .env file in the project root with your credentials:
code
Env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db
DB_USER=your_user
DB_PASSWORD=your_password

# MongoDB
MONGO_URI="enter mongo uri"
MONGO_DB_NAME="enter db name"

# Services
GROQ_API_KEY="your_groq_api_key"
JWT_SECRET_KEY="your_jwt_secret"

***** Run the Server*****

uvicorn darshini_ai:app --reload