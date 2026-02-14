"""
WebSocket Routes for Real-Time Updates

Provides WebSocket endpoints for:
- Real-time workflow status updates
- Event streaming
- System notifications
"""

import sys
import os
import json
from typing import Optional

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.responses import HTMLResponse

from services.api.src.websocket import manager, send_workflow_status_update
from services.api.src.auth.jwt import verify_token
from infrastructure.observability import get_logger, set_trace_id


logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT access token"),
):
    """
    WebSocket endpoint for real-time updates.
    
    Requires authentication via JWT token in query parameter.
    
    **Message Types Received**:
    - `ping`: Keep-alive ping
    - `subscribe`: Subscribe to workflow updates
    - `unsubscribe`: Unsubscribe from workflow updates
    
    **Message Types Sent**:
    - `workflow_status`: Workflow execution status updates
    - `event`: Domain event notifications
    - `notification`: System notifications
    - `pong`: Response to ping
    
    **Example Client Code**:
    ```javascript
    const ws = new WebSocket('ws://localhost:8100/ws/connect?token=YOUR_JWT_TOKEN');
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data.type, data);
    };
    
    // Subscribe to workflow
    ws.send(JSON.stringify({
        action: 'subscribe',
        workflow_id: 'my-workflow'
    }));
    ```
    """
    # Validate token
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    try:
        payload = verify_token(token, token_type="access")
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
    except Exception as e:
        logger.error(f"WebSocket auth failed: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Connect
    await manager.connect(websocket, user_id)
    
    # Send welcome message
    await manager.send_personal_message({
        "type": "connected",
        "message": "WebSocket connection established",
        "user_id": user_id,
    }, websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get("action")
                
                # Handle different actions
                if action == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, websocket)
                
                elif action == "subscribe":
                    workflow_id = message.get("workflow_id")
                    if workflow_id:
                        await manager.subscribe_to_workflow(websocket, workflow_id)
                        await manager.send_personal_message({
                            "type": "subscribed",
                            "workflow_id": workflow_id
                        }, websocket)
                
                elif action == "unsubscribe":
                    workflow_id = message.get("workflow_id")
                    # Unsubscribe logic would go here
                    await manager.send_personal_message({
                        "type": "unsubscribed",
                        "workflow_id": workflow_id
                    }, websocket)
                
                elif action == "get_stats":
                    stats = manager.get_stats()
                    await manager.send_personal_message({
                        "type": "stats",
                        "data": stats
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown action: {action}"
                    }, websocket)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON"
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)


@router.get("/test")
async def websocket_test_page():
    """
    Test page for WebSocket connection.
    
    Provides a simple HTML page to test WebSocket functionality.
    """
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>CognitionOS WebSocket Test</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                }
                #messages {
                    border: 1px solid #ccc;
                    height: 400px;
                    overflow-y: scroll;
                    padding: 10px;
                    margin: 20px 0;
                    background: #f5f5f5;
                }
                .message {
                    padding: 5px;
                    margin: 5px 0;
                    border-radius: 3px;
                }
                .message.received {
                    background: #e3f2fd;
                }
                .message.sent {
                    background: #fff3e0;
                }
                input, button {
                    padding: 10px;
                    margin: 5px;
                }
                .connected {
                    color: green;
                    font-weight: bold;
                }
                .disconnected {
                    color: red;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <h1>CognitionOS WebSocket Test</h1>
            <p>Status: <span id="status" class="disconnected">Disconnected</span></p>
            
            <div>
                <input type="text" id="token" placeholder="JWT Token" style="width: 400px;">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
            </div>
            
            <div>
                <input type="text" id="workflowId" placeholder="Workflow ID">
                <button onclick="subscribe()">Subscribe</button>
                <button onclick="ping()">Ping</button>
                <button onclick="getStats()">Get Stats</button>
            </div>
            
            <div id="messages"></div>
            
            <script>
                let ws = null;
                
                function connect() {
                    const token = document.getElementById('token').value;
                    if (!token) {
                        alert('Please enter JWT token');
                        return;
                    }
                    
                    ws = new WebSocket(`ws://localhost:8100/ws/connect?token=${token}`);
                    
                    ws.onopen = function(event) {
                        document.getElementById('status').textContent = 'Connected';
                        document.getElementById('status').className = 'connected';
                        addMessage('Connected to WebSocket', 'system');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        addMessage(JSON.stringify(data, null, 2), 'received');
                    };
                    
                    ws.onclose = function(event) {
                        document.getElementById('status').textContent = 'Disconnected';
                        document.getElementById('status').className = 'disconnected';
                        addMessage('Disconnected from WebSocket', 'system');
                    };
                    
                    ws.onerror = function(error) {
                        addMessage('Error: ' + error, 'error');
                    };
                }
                
                function disconnect() {
                    if (ws) {
                        ws.close();
                    }
                }
                
                function subscribe() {
                    const workflowId = document.getElementById('workflowId').value;
                    if (!workflowId) {
                        alert('Please enter workflow ID');
                        return;
                    }
                    
                    send({action: 'subscribe', workflow_id: workflowId});
                }
                
                function ping() {
                    send({action: 'ping', timestamp: new Date().toISOString()});
                }
                
                function getStats() {
                    send({action: 'get_stats'});
                }
                
                function send(data) {
                    if (!ws || ws.readyState !== WebSocket.OPEN) {
                        alert('Not connected');
                        return;
                    }
                    
                    ws.send(JSON.stringify(data));
                    addMessage(JSON.stringify(data, null, 2), 'sent');
                }
                
                function addMessage(text, type) {
                    const messagesDiv = document.getElementById('messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + type;
                    messageDiv.textContent = new Date().toLocaleTimeString() + ' - ' + text;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html)
