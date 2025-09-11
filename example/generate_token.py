# generate_token.py
import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY not found in .env file!")

# Create the token payload.
# The user_id can be any integer for testing purposes.
payload = {
    "UserId": 123,  # This is what your `get_validated_token_payload` function looks for
    "exp": datetime.utcnow() + timedelta(days=365) # Token expires in 1 hour
}

# Generate the token
token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

print("Your JWT Bearer Token for Postman:")
print(token)