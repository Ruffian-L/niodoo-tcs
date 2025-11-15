#!/usr/bin/env python3
"""
vLLM Multi-Model Proxy
Routes requests to different vLLM instances based on model name.
"""
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
from typing import Dict, Optional

# Model routing configuration
MODEL_ROUTES: Dict[str, int] = {
    "granite": 8002,
    "granite-3b-code-instruct": 8002,
    "ibm-granite/granite-3b-code-instruct": 8002,
    "qwen": 8003,
    "qwen25-coder-topology": 8003,
    "qwen25-coder-topology-20251105": 8003,
    "curator": 8003,
}

# Default ports
DEFAULT_GRANITE_PORT = 8002
DEFAULT_CURATOR_PORT = 8003
PROXY_PORT = int(os.getenv("VLLM_PROXY_PORT", "8000"))


class VLLMProxyHandler(BaseHTTPRequestHandler):
    def _route_request(self, path: str, method: str, body: Optional[bytes] = None) -> tuple[int, dict, bytes]:
        """Route request to appropriate vLLM instance based on model name."""
        target_port = None
        
        # Parse model name from request
        if method == "GET" and path.startswith("/v1/models"):
            # List models - return both
            return self._list_all_models()
        
        if method == "POST":
            try:
                if body:
                    data = json.loads(body.decode('utf-8'))
                    model_name = data.get('model', '')
                    
                    # Find matching route - check both model name and full path
                    model_lower = model_name.lower()
                    for key, port in MODEL_ROUTES.items():
                        if key.lower() in model_lower or model_lower.startswith(f"granite-{key.lower()}"):
                            target_port = port
                            break
                    
                    # Default routing based on model name patterns
                    if not target_port:
                        if 'granite' in model_lower or model_lower.startswith('granite-'):
                            target_port = DEFAULT_GRANITE_PORT
                        elif 'qwen' in model_lower or 'curator' in model_lower or model_lower.startswith('qwen-curator-'):
                            target_port = DEFAULT_CURATOR_PORT
                        else:
                            # Default to granite
                            target_port = DEFAULT_GRANITE_PORT
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        # If no model specified, check path
        if not target_port:
            if '/granite' in path.lower():
                target_port = DEFAULT_GRANITE_PORT
            elif '/qwen' in path.lower() or '/curator' in path.lower():
                target_port = DEFAULT_CURATOR_PORT
            else:
                # Default to granite for health checks, etc.
                target_port = DEFAULT_GRANITE_PORT
        
        # Forward request
        target_url = f"http://127.0.0.1:{target_port}{path}"
        
        try:
            req = urllib.request.Request(target_url, data=body, method=method)
            # Copy headers, but update content-length if body is present
            for header, value in self.headers.items():
                header_lower = header.lower()
                if header_lower not in ['host']:
                    if header_lower == 'content-length' and body:
                        req.add_header('Content-Length', str(len(body)))
                    else:
                        req.add_header(header, value)
            
            # Add content-length if body exists and header wasn't set
            if body and 'Content-Length' not in req.headers:
                req.add_header('Content-Length', str(len(body)))
            
            with urllib.request.urlopen(req, timeout=300) as response:
                response_body = response.read()
                headers = dict(response.headers)
                return response.status, headers, response_body
        except urllib.error.HTTPError as e:
            # Forward HTTP errors
            error_body = e.read() if hasattr(e, 'read') else b'{}'
            return e.code, dict(e.headers) if hasattr(e, 'headers') else {}, error_body
        except Exception as e:
            self.log_error(f"Error forwarding request to {target_url}: {e}")
            return 500, {"Content-Type": "application/json"}, json.dumps({"error": str(e)}).encode()
    
    def _list_all_models(self) -> tuple[int, dict, bytes]:
        """List all available models from both instances."""
        models = []
        
        # Get models from granite instance
        try:
            with urllib.request.urlopen("http://127.0.0.1:8002/v1/models", timeout=5) as resp:
                granite_data = json.loads(resp.read().decode())
                if 'data' in granite_data:
                    for model in granite_data['data']:
                        model['id'] = f"granite-{model.get('id', 'granite-3b-code-instruct')}"
                        models.append(model)
        except Exception as e:
            self.log_error(f"Error fetching granite models: {e}")
        
        # Get models from curator instance
        try:
            with urllib.request.urlopen("http://127.0.0.1:8003/v1/models", timeout=5) as resp:
                curator_data = json.loads(resp.read().decode())
                if 'data' in curator_data:
                    for model in curator_data['data']:
                        model['id'] = f"qwen-curator-{model.get('id', 'qwen25-coder-topology')}"
                        models.append(model)
        except Exception as e:
            self.log_error(f"Error fetching curator models: {e}")
        
        response_data = {
            "object": "list",
            "data": models
        }
        
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode()
    
    def do_GET(self):
        status, headers, body = self._route_request(self.path, "GET")
        self.send_response(status)
        for header, value in headers.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        status, headers, body = self._route_request(self.path, "POST", body)
        self.send_response(status)
        for header, value in headers.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)
    
    def log_message(self, format, *args):
        """Override to use stderr."""
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")
    
    def log_error(self, message):
        """Log error messages."""
        sys.stderr.write(f"ERROR: {message}\n")


def main():
    server = HTTPServer(("0.0.0.0", PROXY_PORT), VLLMProxyHandler)
    print(f"vLLM Proxy started on port {PROXY_PORT}")
    print(f"Routing:")
    print(f"  Granite models -> http://127.0.0.1:{DEFAULT_GRANITE_PORT}")
    print(f"  Qwen/Curator models -> http://127.0.0.1:{DEFAULT_CURATOR_PORT}")
    sys.stderr.write(f"vLLM Proxy started on port {PROXY_PORT}\n")
    sys.stderr.write(f"Routing: Granite -> {DEFAULT_GRANITE_PORT}, Curator -> {DEFAULT_CURATOR_PORT}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
        server.shutdown()


if __name__ == "__main__":
    main()

