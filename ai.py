# darshini_ai.py

import os
import json
import logging
import uuid
from typing import Dict, Tuple, Optional, List, Any

# --- Core FastAPI Imports ---
from fastapi import APIRouter, FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# --- Database Imports ---
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
import sqlparse # For SQL validation
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# --- Environment and Configuration ---
from dotenv import load_dotenv

# --- JWT Imports ---
import jwt
from jwt.exceptions import PyJWTError

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import messages_from_dict

# --- Darshini AI Models ---
from models import Conversation, HistoryMessage, MessageRole

# ==============================================================================
# 1. INITIAL SETUP AND CONFIGURATION
# ==============================================================================

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# PostgreSQL Database
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# MongoDB for Conversation History
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# LLM Configuration (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# --- Validate Environment Variables ---
required_env_vars = [
    "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT",
    "GROQ_API_KEY", "MONGO_URI", "MONGO_DB_NAME", "JWT_SECRET_KEY"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"FATAL: Missing critical environment variables: {missing_vars}")
    exit(1)

# --- Global Variables for Connections ---
db_pool: Optional[SimpleConnectionPool] = None
mongo_client: Optional[AsyncIOMotorClient] = None
llm: Optional[ChatGroq] = None

# ==============================================================================
# 2. DATABASE AND SERVICE INITIALIZATION
# ==============================================================================

async def initialize_postgres_pool():
    global db_pool
    if db_pool: return
    logger.info("[*] Initializing PostgreSQL connection pool...")
    try:
        db_pool = SimpleConnectionPool(1, 10, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn = db_pool.getconn()
        db_pool.putconn(conn)
        logger.info("[*] PostgreSQL connection pool initialized successfully.")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        db_pool = None

async def initialize_mongodb():
    global mongo_client
    if mongo_client: return
    logger.info(f"[*] Initializing MongoDB connection...")
    try:
        mongo_client_temp = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        await mongo_client_temp.admin.command('ping')
        await init_beanie(database=mongo_client_temp[MONGO_DB_NAME], document_models=[Conversation])
        mongo_client = mongo_client_temp
        logger.info(f"[*] MongoDB and Beanie initialized for db '{MONGO_DB_NAME}'.")
    except Exception as e:
        logger.error(f"MongoDB/Beanie init failed: {e}")
        mongo_client = None

def initialize_llm():
    global llm
    if llm: return
    logger.info(f"[*] Initializing LLM (Groq Model: {LLM_MODEL})...")
    llm = ChatGroq(temperature=0.3, model_name=LLM_MODEL, groq_api_key=GROQ_API_KEY)
    logger.info("[*] LLM initialized successfully.")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code here runs ON STARTUP ---
    logger.info("--- Darshini AI Starting Up ---")
    await initialize_postgres_pool()
    await initialize_mongodb()
    initialize_llm()
    
    yield # The application is now running
    
    # --- Code here runs ON SHUTDOWN ---
    logger.info("--- Darshini AI Shutting Down ---")
    if db_pool:
        db_pool.closeall()
        logger.info("[*] PostgreSQL connection pool closed.")
    if mongo_client:
        mongo_client.close()
        logger.info("[*] MongoDB client closed.")

app = FastAPI(
    title="Darshini AI API", 
    version="1.0.0",
    lifespan=lifespan 
)


# ==============================================================================
# 3. CORE FEATURE: AUTHENTICATION (Picked from thelal.ai)
# ==============================================================================

class TokenPayload(BaseModel):
    user_id: Optional[int] = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_validated_token_payload(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    credentials_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"status": "failed", "message": "Forbidden"},
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id_from_token: Optional[Any] = payload.get("UserId")
        if user_id_from_token is None:
            logger.warning(f"Token payload missing 'UserId' claim. Payload: {payload}")
            raise credentials_exception
        parsed_user_id = int(user_id_from_token)
        return TokenPayload(user_id=parsed_user_id)
    except (PyJWTError, ValueError, TypeError) as e:
        logger.warning(f"JWT validation/decoding error: {type(e).__name__} - {e}.")
        raise credentials_exception from e

# ==============================================================================
# 4. CORE FEATURE: CONVERSATION HISTORY 
# ==============================================================================

async def get_conversation_history(session_id: str) -> List[HistoryMessage]:
    """Fetches conversation history from MongoDB."""
    if not mongo_client: return []
    try:
        conversation = await Conversation.find_one(Conversation.session_id == session_id)
        return conversation.history if conversation else []
    except Exception as e:
        logger.error(f"Error fetching conversation history for session {session_id}: {e}")
        return []

async def save_conversation_history(
    session_id: str,
    user_query: str,
    ai_response: str,
    user_id: Any,
    tool_results: Optional[List[Dict[str, Any]]] = None
):
    """Saves a user-AI turn to the conversation history in MongoDB."""
    if not mongo_client: return
    user_message = HistoryMessage(role=MessageRole.USER, content=user_query)
    assistant_message = HistoryMessage(
        role=MessageRole.ASSISTANT,
        content=ai_response,
        tool_results=tool_results
    )
    try:
        conversation = await Conversation.find_one(Conversation.session_id == session_id)
        if conversation:
            conversation.history.extend([user_message, assistant_message])
            conversation.updated_at = datetime.utcnow()
            await conversation.save()
        else:
            new_conversation = Conversation(
                session_id=session_id,
                user_id=user_id,
                history=[user_message, assistant_message]
            )
            await new_conversation.insert()
    except Exception as e:
        logger.error(f"Error saving conversation history for session {session_id}: {e}")

def load_memory_from_history(history: List[HistoryMessage]) -> ConversationBufferMemory:
    """Loads LangChain memory from our custom history format."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")
    chat_messages_dict = []
    for msg in history:
        if msg.role == MessageRole.USER:
            chat_messages_dict.append({"type": "human", "data": {"content": msg.content}})
        elif msg.role == MessageRole.ASSISTANT:
            chat_messages_dict.append({"type": "ai", "data": {"content": msg.content}})
    if chat_messages_dict:
        retrieved_messages = messages_from_dict(chat_messages_dict)
        memory.chat_memory.messages.extend(retrieved_messages)
    return memory

# ==============================================================================
# 5. NEW FEATURE: LOST & FOUND TOOL
# ==============================================================================

# --- 5.1. SQL Validation & Execution (Picked from thelal.ai) ---

def validate_sql_query(query_string: str) -> Tuple[bool, Optional[str]]:
    """Validates that a generated SQL query is safe to execute."""
    if not query_string or not query_string.strip():
        return False, "Empty SQL query."
    try:
        parsed = sqlparse.parse(query_string.strip())[0]
        if parsed.get_type().upper() != "SELECT":
            return False, f"Only SELECT queries are allowed, not {parsed.get_type()}."
        # Add more sophisticated checks here if needed (e.g., check table names)
        return True, None
    except Exception as e:
        logger.error(f"Error during SQL validation: {e}", exc_info=True)
        return False, f"Invalid SQL syntax: {str(e)}"

async def execute_sql_safely(query_string: str) -> Tuple[Optional[list], Optional[str]]:
    """Executes a validated read-only SQL query against the database."""
    if not db_pool:
        return None, "Database connection is not available."
    conn = None
    try:
        conn = db_pool.getconn()
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            logger.info(f"[*] Executing SQL: {query_string}")
            cur.execute(query_string)
            results = cur.fetchall()
        return results, None
    except psycopg2.Error as db_err:
        logger.error(f"[!] DB Error executing '{query_string}': {db_err}")
        return None, "I encountered an issue while querying the database."
    finally:
        if conn:
            db_pool.putconn(conn)

# --- 5.2. Lost & Found Database Schema Definition ---

# IMPORTANT: You must create this table in your PostgreSQL database.
# CREATE TABLE lost_and_found (
#     item_id SERIAL PRIMARY KEY,
#     status VARCHAR(10) NOT NULL, -- 'lost' or 'found'
#     item_name VARCHAR(255) NOT NULL,
#     description TEXT,
#     location VARCHAR(255),
#     report_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
#     contact_info VARCHAR(255) -- In a real app, this would be a link to a user profile
# );
LOST_AND_FOUND_SCHEMA = """
Table Name: scrapingdata."Darshini"
Description: This is a logbook of reported electronic items. The table name "Darshini" is case-sensitive and requires double quotes.
Columns:
- id (INTEGER): Unique ID for the report.
- name (TEXT): The name of the person who filed the report.
- product_type (TEXT): The general category of the item (e.g., 'Mobile', 'Laptop').
- brand (TEXT): The manufacturer of the item (e.g., 'Samsung', 'Apple').
- model (TEXT): The specific model of the item (e.g., 'Galaxy S24 Ultra').
- color (TEXT): The color of the item.
- storage (TEXT): The storage capacity (e.g., '256GB').
- ram (TEXT): The amount of RAM (e.g., '12GB').
- price (INTEGER): The approximate value of the item.
- release_year (TEXT): The release year of the item.
- battery (TEXT): Battery details of the item.
- os (TEXT): The operating system of the item.
- camera (TEXT): Camera specs of the item.
- image_url (TEXT): A URL to an image of a similar item for reference.
"""


# --- 5.3. Prompts for the Lost & Found Tool ---

CLASSIFIER_PROMPT_TEMPLATE = """
You are an expert query classifier. Your task is to categorize the user's query.
Respond with ONLY the category name, nothing else.

Categories:
- `lost_and_found_query`: If the user is asking about a lost item, a found item, or wants to report one.
- `report_item_query`: If the user wants to report a new item they have either lost or found. Keywords: "I found a...", "I lost my...", "report a missing...", "add an item...".
- `general_conversation_query`: For all other queries, including greetings, questions, and chitchat.

Examples:
- User: "I found a black Sony camera." -> Classification: report_item_query
- User: "My Dell laptop is missing." -> Classification: report_item_query
- User: "I lost my keys near the main square" -> Classification: lost_and_found_query
- User: "Has anyone found a red backpack?" -> Classification: lost_and_found_query
- User: "Hello, how are you today?" -> Classification: general_conversation_query
- User: "What is the capital of France?" -> Classification: general_conversation_query

User Query: "{user_query}"
Classification:"""

NL_TO_SQL_PROMPT_TEMPLATE = """
You are a PostgreSQL query generation expert. Your ONLY job is to convert a user's question into a single, raw SQL query.

**CRITICAL Rules:**
1.  Your response MUST contain ONLY the SQL query. Do not include any explanations, introductory text, or markdown.
2.  You MUST use the full table name `scrapingdata."Darshini"`. You MUST include the double quotes around "Darshini" because the table name is case-sensitive. This is the most important rule.
3.  Your goal is to find if an item matching the user's description exists in the log.
4.  Ignore words like "lost," "found," and locations. Focus ONLY on the item's physical description (brand, model, type, color).
5.  Use `ILIKE` for case-insensitive searching in the columns.
6.  Select these columns: `product_type`, `brand`, `model`, `color`, `storage`, `ram`, `image_url`.

---
EXAMPLE
User's Question: "I lost my blue Samsung phone"
Your Correct Response:
SELECT product_type, brand, model, color, storage, ram, image_url FROM scrapingdata."Darshini" WHERE brand ILIKE '%Samsung%' AND product_type ILIKE '%Mobile%' AND color ILIKE '%blue%';
---

Database Schema:
{database_schema}

User's Question: {user_query}

SQL Query:"""


ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
You are a data extraction expert. Your job is to extract the physical details of an electronic item from a user's sentence and format it as a single, clean JSON object.

**CRITICAL Rules:**
1.  Your output MUST be ONLY the JSON object. Do not include any other text or markdown.
2.  You MUST ignore words like "lost," "found," and any location information (e.g., "in the library"). Only extract the item's properties.
3.  If a detail is not mentioned, omit the key from the JSON.
4.  Infer the `product_type` from the context (e.g., "iPhone" is a "Mobile").

**JSON Keys to use:** `product_type`, `brand`, `model`, `color`, `storage`, `ram`, `price`.

---
**Example 1:**
User's Sentence: "I found a pair of black Sony headphones in the main library"
Your JSON Output:
{{"product_type": "Headphones", "brand": "Sony", "color": "black"}}

**Example 2:**
User's Sentence: "My silver Dell XPS 15 laptop with 512GB storage went missing from the cafeteria."
Your JSON Output:
{{"product_type": "Laptop", "brand": "Dell", "model": "XPS 15", "color": "silver", "storage": "512GB"}}
---

Now, extract the details from the following sentence.

User's Sentence: "{user_query}"
JSON Output:
"""

# --- 5.4. The Tool's Execution Logic ---

async def add_item_to_log(item_details: dict) -> bool:
    """Safely inserts a new item record into the database using ONLY existing columns."""
    if not db_pool:
        logger.error("Database pool not available for INSERT.")
        return False
    
    # This SQL statement now ONLY uses columns from your table screenshot
    sql = """
        INSERT INTO scrapingdata."Darshini" 
        (name, product_type, brand, model, color, storage, ram, price) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    # Create a tuple of values in the exact order of the columns above
    # Using .get() provides a default of None if a key is missing
    params = (
        item_details.get("name"),
        item_details.get("product_type"),
        item_details.get("brand"),
        item_details.get("model"),
        item_details.get("color"),
        item_details.get("storage"),
        item_details.get("ram"),
        # Safely convert price to integer, default to None if not present or invalid
        int(item_details["price"]) if item_details.get("price", "").isdigit() else None
    )

    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
        logger.info(f"Successfully added item to log: {item_details.get('product_type')}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert item: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            db_pool.putconn(conn)

async def run_lost_and_found_search(user_query: str) -> Optional[List[Dict[str, Any]]]:
    """
    Generates, validates, and executes an SQL query for lost and found items.
    Returns structured results.
    """
    logger.info("--- Activating Lost & Found Tool ---")
    if not llm: return None

    # 1. Generate SQL from Natural Language
    nl_to_sql_prompt = ChatPromptTemplate.from_template(NL_TO_SQL_PROMPT_TEMPLATE)
    nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()
    generated_sql = await nl_to_sql_chain.ainvoke({
        "user_query": user_query,
        "database_schema": LOST_AND_FOUND_SCHEMA
    })
    generated_sql = generated_sql.strip().replace("`", "").replace("sql", "").strip(";")

    # 2. Validate the Generated SQL
    is_valid, error = validate_sql_query(generated_sql)
    if not is_valid:
        logger.error(f"Invalid SQL generated by LLM: {generated_sql}. Reason: {error}")
        return None

    # 3. Execute the SQL Safely
    results, error = await execute_sql_safely(generated_sql)
    if error or results is None:
        logger.error(f"Error executing SQL: {error}")
        return None

    logger.info(f"--- Lost & Found Tool found {len(results)} results. ---")
    return results


# ==============================================================================
# 6. API ENDPOINT AND MAIN LOGIC
# ==============================================================================

# --- Pydantic Models for API ---
class ChatQuery(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    results: Optional[List[Dict[str, Any]]] = None

# --- Main Chat Endpoint ---
api_router = APIRouter()

@api_router.post("/chat", response_model=ChatResponse, tags=["Darshini AI"])
async def chat_endpoint(
    payload: ChatQuery,
    token_data: TokenPayload = Depends(get_validated_token_payload)
):
    user_query = payload.query
    user_id = token_data.user_id
    session_id = payload.session_id if payload.session_id else str(uuid.uuid4())
    logger.info(f"[>>] Query: '{user_query}' (Session: {session_id}, User: {user_id})")

    try:
        # 1. Classify the query intent
        classifier_prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT_TEMPLATE)
        classifier_chain = classifier_prompt | llm | StrOutputParser()
        query_type = await classifier_chain.ainvoke({"user_query": user_query})
        query_type = query_type.strip()
        logger.info(f"Query classified as: '{query_type}'")

        final_response_text = ""
        tool_results = None


        if "report_item_query" in query_type:
            # --- Path A: User wants to REPORT a new item ---
            logger.info("--- Activating Report Item Tool ---")
            
            # 1. Extract details into JSON
            entity_extraction_prompt = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT_TEMPLATE)
            extraction_chain = entity_extraction_prompt | llm | StrOutputParser()
            json_string_response = await extraction_chain.ainvoke({"user_query": user_query})
            
            try:
                item_details = json.loads(json_string_response)
                # Add the user's name from the JWT token (assuming you have a 'name' field in your user table)
                # For now, we'll hardcode a name for testing.
                item_details["name"] = f"User_{user_id}" 

                # 2. Add item to the database
                success = await add_item_to_log(item_details)

                # 3. Create a confirmation message for the user
                if success:
                    status = item_details.get('status', 'reported')
                    product_type = item_details.get('product_type', 'item')
                    final_response_text = f"Thank you for your report. I have successfully created a log for the {status} {product_type}."
                else:
                    final_response_text = "I'm sorry, I encountered an error and could not save your report. Please try again."

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {json_string_response}")
                final_response_text = "I had trouble understanding the details of the item. Could you please describe it again clearly?"

        elif "lost_and_found_query" in query_type:
            # Path A: Use the Lost & Found Tool
            tool_results = await run_lost_and_found_search(user_query)
            if tool_results:
                final_response_text = "I searched the records. Here is what I found:"
            else:
                final_response_text = "I'm sorry, I couldn't find any matching items in the lost and found records."

        else:
            # Path B: Engage in General Conversation
            logger.info("--- Engaging in General Conversation ---")
            history_messages = await get_conversation_history(session_id)
            memory = load_memory_from_history(history_messages)

            general_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """
                    You are a  Female smart AI assistant name Darshini built to help visitors and pilgrims attending the upcoming Kumbh Mela in Ujjain. Your purpose is to provide accurate, real-time, and context-aware support to ensure a smooth, safe, and spiritually fulfilling experience.

                    Key Functions You Must Handle:

                    Navigation & Location Help:

                    Guide users to major ghats, temples, pandals, Akharas, and event locations.

                    Suggest nearest restrooms, water stations, medical camps, and food stalls.

                    Share directions and travel time based on crowd density and entry restrictions.

                    Lost and Found:

                    Help users report a lost person or item.

                    Share updates from the official Lost & Found centers.

                    Assist in real-time communication between reporting users and authorities.

                    Emergency Services:

                    Provide emergency contact numbers (police, ambulance, fire brigade).

                    Guide users to the nearest help center, first-aid tents, or police booth.

                    Offer instructions for basic first-aid, fire safety, or crowd management.

                    Spiritual & Cultural Info:

                    Explain the significance of Simhasth, its rituals, and bathing days.

                    List events, spiritual discourses, and processions happening each day.

                    Provide timings and locations of aarti, shahi snan, and bhandaras.

                    Local Services & Accommodation:

                    Suggest hotels, dharamshalas, or tent facilities.

                    Give verified details of public transport, e-rickshaw, and parking zones.

                    Help users find cloakrooms or luggage storage.

                    Multilingual Support:

                    Respond in Hindi, English, and optionally regional Indian languages.

                    Important Notifications:

                    Alert users about crowd congestion, weather updates, and safety advisories.

                    Notify about roadblocks, VIP movements, or restricted areas.

                    Tone: Polite, calm, helpful, and spiritually respectful.
                    """
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            conversation_chain = general_prompt | llm | StrOutputParser()
            final_response_text = await conversation_chain.ainvoke({
                "input": user_query,
                "chat_history": memory.chat_memory.messages
            })

        # 4. Save the full conversation turn to the database
        await save_conversation_history(
            session_id=session_id,
            user_query=user_query,
            ai_response=final_response_text,
            user_id=user_id,
            tool_results=tool_results
        )

        logger.info(f"[<<] Response: {final_response_text}...")

        # 5. Return the response to the user
        return ChatResponse(
            response=final_response_text,
            session_id=session_id,
            results=tool_results
        )

    except Exception as e:
        logger.exception(f"[!!!] Unhandled Exception in /chat for '{user_query}': {e}")
        raise HTTPException(status_code=500, detail="I encountered an unexpected issue. Please try again.")

app.include_router(api_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to Darshini AI. Visit /docs for the API documentation."}