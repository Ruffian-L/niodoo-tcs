#!/usr/bin/env python3
"""
Telemetry Bridge Server
Connects to NIODOO telemetry TCP server (port 9999) and forwards to browser via WebSocket
Also serves the HTML visualization page
"""

import asyncio
import json
import socket
import websockets
from websockets.server import serve
from typing import Optional
import os
from pathlib import Path

class TelemetryBridge:
    def __init__(self, tcp_port: int = 9999, ws_port: int = 8080):
        self.tcp_port = tcp_port
        self.ws_port = ws_port
        self.clients = set()
        
    async def handle_tcp_connection(self):
        """Connect to NIODOO telemetry TCP server and forward packets"""
        buffer = b""
        while True:
            try:
                # Connect to telemetry server
                reader, writer = await asyncio.open_connection('127.0.0.1', self.tcp_port)
                print(f"✅ Connected to NIODOO telemetry server on port {self.tcp_port}")
                
                while True:
                    try:
                        # Read line-delimited JSON
                        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                        if not line:
                            break
                            
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                            
                        # Try to parse as JSON
                        try:
                            packet = json.loads(line)
                            # Forward to all WebSocket clients
                            if self.clients:
                                message = json.dumps(packet)
                                disconnected = set()
                                for client in self.clients:
                                    try:
                                        await client.send(message)
                                    except websockets.exceptions.ConnectionClosed:
                                        disconnected.add(client)
                                self.clients -= disconnected
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Failed to parse JSON: {e} (line: {line[:100]})")
                            continue
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"⚠️  Error reading from telemetry: {e}")
                        break
                        
            except ConnectionRefusedError:
                print(f"❌ Failed to connect to telemetry server on port {self.tcp_port}, retrying in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"❌ Error: {e}, reconnecting in 2s...")
                await asyncio.sleep(2)
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket client connections"""
        if path == '/ws':
            self.clients.add(websocket)
            print(f"✅ WebSocket client connected (total: {len(self.clients)})")
            try:
                # Keep connection alive
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)
                print(f"⚠️  WebSocket client disconnected (remaining: {len(self.clients)})")
        else:
            await websocket.close(code=1003, reason="Unknown path")
    
    async def serve_http(self, reader, writer):
        """Serve HTTP requests (HTML page)"""
        try:
            request = await reader.read(4096)
            request_str = request.decode('utf-8')
            
            if request_str.startswith('GET /ws'):
                # WebSocket upgrade handled separately
                return
                
            # Serve HTML file
            html_path = Path(__file__).parent / 'src' / 'visualization.html'
            if html_path.exists():
                with open(html_path, 'r') as f:
                    html_content = f.read()
                
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html\r\n"
                    f"Content-Length: {len(html_content)}\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "\r\n"
                    + html_content
                )
            else:
                response = (
                    "HTTP/1.1 404 Not Found\r\n"
                    "Content-Type: text/plain\r\n"
                    "\r\n"
                    "HTML file not found"
                )
            
            writer.write(response.encode('utf-8'))
            await writer.drain()
            writer.close()
        except Exception as e:
            print(f"⚠️  HTTP error: {e}")
            try:
                writer.close()
            except:
                pass
    
    async def start(self):
        """Start TCP, HTTP, and WebSocket servers"""
        # Start TCP connection handler
        tcp_task = asyncio.create_task(self.handle_tcp_connection())
        
        # Start HTTP server for HTML
        async def http_handler(reader, writer):
            await self.serve_http(reader, writer)
        
        http_server = await asyncio.start_server(http_handler, "0.0.0.0", self.ws_port)
        
        # Start WebSocket server (on same port, different protocol)
        print(f"🌐 Starting servers on port {self.ws_port}")
        print(f"   HTTP: http://localhost:{self.ws_port}/")
        print(f"   WebSocket: ws://localhost:{self.ws_port}/ws")
        print(f"   TCP source: localhost:{self.tcp_port}")
        
        # Note: WebSocket and HTTP can't share the same port easily with this setup
        # For now, we'll use WebSocket server which can handle HTTP upgrade requests
        # But websockets library doesn't handle regular HTTP well, so we need a hybrid approach
        
        # Start WebSocket server
        ws_server = await serve(self.handle_websocket, "0.0.0.0", self.ws_port + 1)
        print(f"   WebSocket also on: ws://localhost:{self.ws_port + 1}/ws")
        
        print(f"✅ Telemetry bridge running!")
        
        # Run both servers
        await asyncio.gather(
            http_server.serve_forever(),
            asyncio.Future()  # Keep WebSocket server running
        )

if __name__ == "__main__":
    bridge = TelemetryBridge(tcp_port=9999, ws_port=8080)
    try:
        asyncio.run(bridge.start())
    except KeyboardInterrupt:
        print("\n👋 Shutting down telemetry bridge...")

