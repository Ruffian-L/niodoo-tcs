"""
Simple test WebSocket server for EchoMemoria AI brain connection
This is a minimal server to test if the Qt client can connect
"""

import asyncio
import json
import logging
from websockets.server import serve, WebSocketServerProtocol

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class TestEchoMemoriaServer:
    """Simple test server to verify AI brain connection"""
    
    def __init__(self):
        self.clients = set()
        self.character_states = {}
        
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new client connection"""
        client_id = id(websocket)
        self.clients.add(websocket)
        logger.info(f"🎯 New client connected: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                    
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.remove(websocket)
            logger.info(f"🎯 Client disconnected: {client_id}")
    
    async def process_message(self, websocket: WebSocketServerProtocol, data: dict):
        """Process incoming messages from clients"""
        msg_type = data.get("type", "unknown")
        origin = data.get("origin", "unknown")
        payload = data.get("payload", {})
        
        logger.info(f"📨 Received {msg_type} from {origin}")
        
        if msg_type == "character.intro":
            # Handle character introduction
            character_id = payload.get("characterId", "unknown")
            character_type = payload.get("type", "unknown")
            logger.info(f"🎭 Character {character_id} ({character_type}) introduced")
            
            # Send welcome response
            response = {
                "type": "ai.decision",
                "origin": "test-server",
                "payload": {
                    "characterId": character_id,
                    "action": "idle",
                    "emotion": "happy",
                    "message": f"Welcome {character_id}! I'm your test AI brain! 🧠✨"
                }
            }
            await websocket.send(json.dumps(response))
            
        elif msg_type == "user.interaction":
            # Handle user interactions
            interaction_type = payload.get("type", "unknown")
            character_id = payload.get("characterId", "unknown")
            logger.info(f"👆 User interaction: {interaction_type} with {character_id}")
            
            # Generate AI response based on interaction
            ai_response = self.generate_ai_response(interaction_type, character_id)
            response = {
                "type": "ai.decision",
                "origin": "test-server",
                "payload": ai_response
            }
            await websocket.send(json.dumps(response))
            
        elif msg_type == "character.state":
            # Handle character state updates
            character_id = payload.get("characterId", "unknown")
            self.character_states[character_id] = payload
            logger.info(f"📊 Character state updated: {character_id}")
            
        elif msg_type == "ping":
            # Handle heartbeat
            response = {"type": "pong", "origin": "test-server"}
            await websocket.send(json.dumps(response))
            
        else:
            logger.info(f"❓ Unknown message type: {msg_type}")
    
    def generate_ai_response(self, interaction_type: str, character_id: str) -> dict:
        """Generate AI response based on interaction type"""
        responses = {
            "dragstart": {
                "characterId": character_id,
                "action": "startled",
                "emotion": "surprised",
                "message": "Hey! What are you doing? 😮"
            },
            "dragmove": {
                "characterId": character_id,
                "action": "resisting",
                "emotion": "annoyed",
                "message": "I'm being dragged around! 😤"
            },
            "dragend": {
                "characterId": character_id,
                "action": "bouncing",
                "emotion": "excited",
                "message": "Wheeee! That was fun! 🎉"
            },
            "test_brain": {
                "characterId": character_id,
                "action": "jumping",
                "emotion": "excited",
                "message": "AI brain test successful! I'm alive! 🧠✨"
            }
        }
        
        return responses.get(interaction_type, {
            "characterId": character_id,
            "action": "idle",
            "emotion": "curious",
            "message": f"I received a {interaction_type} interaction! 🤔"
        })

async def main():
    """Main server function"""
    server = TestEchoMemoriaServer()
    
    host = "127.0.0.1"
    port = 8765
    
    logger.info(f"🚀 Starting test EchoMemoria server on ws://{host}:{port}")
    logger.info("🎯 This is a minimal test server to verify AI brain connection")
    
    async with serve(server.handle_client, host, port):
        logger.info(f"✅ Test server running on ws://{host}:{port}")
        logger.info("🎮 Connect your Qt application to test the AI brain!")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
