#!/usr/bin/env python3
"""
XenoAI MCP Server
Wraps the XenoAI Flask API so Claude can chat with XenoAI,
manage conversations, and control AI modes.
"""

import json
import os
from typing import Optional
import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# ── Config ────────────────────────────────────────────────────────────────────
XENOAI_BASE_URL = os.environ.get("XENOAI_BASE_URL", "https://9bb.up.railway.app").rstrip("/")
XENOAI_USERNAME = os.environ.get("XENOAI_USERNAME", "")
XENOAI_PASSWORD = os.environ.get("XENOAI_PASSWORD", "")
REQUEST_TIMEOUT = 60.0

mcp = FastMCP("xenoai_mcp")

# ── Shared HTTP client ────────────────────────────────────────────────────────
_session_cookies: dict = {}

async def _request(method: str, path: str, **kwargs) -> dict:
    """Make an authenticated request to XenoAI."""
    url = f"{XENOAI_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, cookies=_session_cookies) as client:
        resp = await client.request(method, url, **kwargs)
        # Persist session cookies
        _session_cookies.update(dict(resp.cookies))
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp
            )
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}

def _handle_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 401: return "Error: Not authenticated. Use xenoai_login first."
        if code == 404: return "Error: Resource not found. Check chat_id."
        if code == 429: return "Error: Rate limited. Wait a moment and retry."
        return f"Error: XenoAI returned HTTP {code}: {e.response.text[:200]}"
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. XenoAI may be waking up — try again in 10 seconds."
    return f"Error: {type(e).__name__}: {str(e)}"

# ── Input Models ──────────────────────────────────────────────────────────────

class LoginInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    username: str = Field(..., description="XenoAI username or email", min_length=1)
    password: str = Field(..., description="XenoAI password", min_length=1)

class ChatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    message: str = Field(..., description="Message to send to XenoAI", min_length=1, max_length=8000)
    chat_id: Optional[str] = Field(default=None, description="Existing chat ID to continue. Omit to start a new chat.")
    mode: Optional[str] = Field(default=None, description="AI mode/persona (e.g. 'code', 'default'). Omit for default.")

class ChatIdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    chat_id: str = Field(..., description="The chat ID to retrieve or delete", min_length=1)

class ListChatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=20, description="Max chats to return", ge=1, le=100)

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="xenoai_login",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
)
async def xenoai_login(params: LoginInput) -> str:
    """Log in to XenoAI. Required before most other operations.

    Args:
        params (LoginInput):
            - username (str): XenoAI username or email
            - password (str): XenoAI password

    Returns:
        str: Success message or error details
    """
    try:
        data = await _request("POST", "/login", json={
            "username": params.username,
            "password": params.password
        })
        return f"✅ Logged in successfully as '{params.username}'"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_chat",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def xenoai_chat(params: ChatInput) -> str:
    """Send a message to XenoAI and get an AI response.
    XenoAI uses a Gemini → Groq Llama 70B → DeepSeek R1 pipeline for build requests.

    Args:
        params (ChatInput):
            - message (str): The user message to send
            - chat_id (Optional[str]): Continue an existing chat, or omit for new chat
            - mode (Optional[str]): AI mode/persona to use

    Returns:
        str: JSON with keys: reply, chat_id, title, pipeline info
    """
    try:
        body: dict = {"message": params.message}
        if params.chat_id:
            body["chat_id"] = params.chat_id
        if params.mode:
            body["mode"] = params.mode

        data = await _request("POST", "/chat", json=body)

        reply = data.get("reply", "")
        chat_id = data.get("chat_id", "")
        title = data.get("title", "New Chat")

        return json.dumps({
            "chat_id": chat_id,
            "title": title,
            "reply": reply,
            "saved_files": data.get("saved_files", []),
            "workspace": data.get("workspace", "")
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_list_chats",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
)
async def xenoai_list_chats(params: ListChatsInput) -> str:
    """List all XenoAI chat conversations.

    Args:
        params (ListChatsInput):
            - limit (int): Max number of chats to return (default 20)

    Returns:
        str: JSON list of chats with id, title, created, message count
    """
    try:
        data = await _request("GET", "/chats")
        chats = data if isinstance(data, list) else data.get("chats", [])
        chats = chats[:params.limit]

        formatted = []
        for c in chats:
            msgs = c.get("messages", [])
            formatted.append({
                "chat_id": c.get("id", ""),
                "title": c.get("title", "New Chat"),
                "mode": c.get("mode", "default"),
                "message_count": len(msgs),
                "created": c.get("created", 0)
            })

        return json.dumps({"total": len(formatted), "chats": formatted}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_get_chat",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
)
async def xenoai_get_chat(params: ChatIdInput) -> str:
    """Get the full message history of a specific XenoAI chat.

    Args:
        params (ChatIdInput):
            - chat_id (str): The chat ID to retrieve

    Returns:
        str: JSON with chat metadata and full message history
    """
    try:
        data = await _request("GET", f"/chat/{params.chat_id}")
        chat = data if isinstance(data, dict) else {}
        messages = chat.get("messages", [])

        # Format messages for readability
        formatted_msgs = []
        for m in messages:
            formatted_msgs.append({
                "role": m.get("role", "unknown"),
                "content": m.get("content", "")[:500] + ("..." if len(m.get("content","")) > 500 else "")
            })

        return json.dumps({
            "chat_id": chat.get("id", params.chat_id),
            "title": chat.get("title", ""),
            "mode": chat.get("mode", "default"),
            "message_count": len(messages),
            "messages": formatted_msgs
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_delete_chat",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False}
)
async def xenoai_delete_chat(params: ChatIdInput) -> str:
    """Delete a specific XenoAI chat conversation. This is irreversible.

    Args:
        params (ChatIdInput):
            - chat_id (str): The chat ID to delete

    Returns:
        str: Success or error message
    """
    try:
        await _request("DELETE", f"/chat/{params.chat_id}")
        return f"✅ Chat '{params.chat_id}' deleted successfully."
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_list_modes",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
)
async def xenoai_list_modes() -> str:
    """List all available AI modes and skills in XenoAI.

    Returns:
        str: JSON list of available modes/personas
    """
    try:
        data = await _request("GET", "/modes")
        modes = data if isinstance(data, list) else data.get("modes", [])
        return json.dumps({"modes": modes, "count": len(modes)}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="xenoai_status",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
)
async def xenoai_status() -> str:
    """Check if XenoAI is online and get basic server info.

    Returns:
        str: Status info including uptime and model info
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(XENOAI_BASE_URL)
            online = resp.status_code < 500
        return json.dumps({
            "status": "online" if online else "degraded",
            "url": XENOAI_BASE_URL,
            "http_code": resp.status_code
        }, indent=2)
    except httpx.TimeoutException:
        return json.dumps({"status": "sleeping", "message": "XenoAI is waking up. Try again in 15 seconds."})
    except Exception as e:
        return json.dumps({"status": "offline", "error": str(e)})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"🤖 XenoAI MCP Server → http://0.0.0.0:{port}")
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
