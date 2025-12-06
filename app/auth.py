"""
Authentication utilities for JWT token extraction
"""
import os
from typing import Optional
from fastapi import HTTPException, Depends, Header
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")


def get_user_id_from_token(authorization: Optional[str] = Header(None, alias="Authorization")) -> str:
    """
    Extract user_id from JWT token in Authorization header.
    
    Args:
        authorization: Authorization header value (format: "Bearer <token>")
        
    Returns:
        user_id: User ID from the JWT token
        
    Raises:
        HTTPException: If token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing. Please include 'Authorization: Bearer <token>' in your request headers.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authorization scheme")
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected: Bearer <token>"
        )
    
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_JWT_SECRET not configured. Please add SUPABASE_JWT_SECRET to your .env file. Get it from Supabase Dashboard > Project Settings > API > JWT Secret"
        )
    
    try:
        # Decode JWT token
        # Supabase tokens may have non-standard audience claims, so we skip audience verification
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False, "verify_signature": True}  # Skip audience check but verify signature
        )
        
        # Extract user_id from payload
        # Supabase typically stores it as 'sub' or 'user_id'
        user_id = payload.get("sub") or payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="User ID not found in token"
            )
        
        return user_id
        
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token validation failed: {str(e)}"
        )

