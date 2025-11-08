# 🌐 REST API Reference

**Created by Jason Van Pham | Niodoo Framework | 2025**

## Overview

The Niodoo-Feeling consciousness engine provides a comprehensive REST API for interacting with the consciousness system. The API follows RESTful principles and supports JSON request/response formats.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

All API endpoints require authentication using Bearer tokens:

```http
Authorization: Bearer <your-token>
```

## Content Type

All requests and responses use JSON:

```http
Content-Type: application/json
Accept: application/json
```

## Core Endpoints

### Consciousness Engine

#### Process Consciousness Event

Process a consciousness event through the entire pipeline.

```http
POST /consciousness/process
```

**Request Body:**
```json
{
  "input": "Hello, world!",
  "emotional_context": {
    "emotion": "joy",
    "intensity": 0.8
  },
  "options": {
    "include_reasoning_trace": true,
    "include_memory_references": true,
    "timeout_ms": 5000
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response_text": "Hello! I'm excited to interact with you.",
    "emotional_context": {
      "emotion": "joy",
      "intensity": 0.9
    },
    "confidence_score": 0.85,
    "reasoning_trace": [
      "Analyzed input for emotional content",
      "Retrieved relevant memories",
      "Applied Möbius transformation",
      "Generated consensus from three-brain system"
    ],
    "memory_references": [
      {
        "id": "mem_123",
        "content": "Previous greeting interaction",
        "relevance": 0.7
      }
    ],
    "consciousness_metrics": {
      "consciousness_level": 0.75,
      "attention_focus": 0.8,
      "memory_coherence": 0.9
    },
    "processing_time_ms": 245
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "PROCESSING_ERROR",
    "message": "Failed to process consciousness event",
    "details": "Brain coordination timeout"
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Get Consciousness State

Retrieve the current consciousness state.

```http
GET /consciousness/state
```

**Response:**
```json
{
  "success": true,
  "data": {
    "consciousness_level": 0.75,
    "attention_focus": 0.8,
    "memory_coherence": 0.9,
    "emotional_context": {
      "joy": 0.6,
      "sadness": 0.1,
      "anger": 0.0,
      "fear": 0.0,
      "curiosity": 0.8
    },
    "active_personalities": [
      {
        "type": "analytical",
        "weight": 0.1
      },
      {
        "type": "creative",
        "weight": 0.15
      }
    ],
    "last_updated": "2025-01-27T10:30:00Z"
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Update Emotional Context

Update the emotional context of the consciousness engine.

```http
PUT /consciousness/emotional-context
```

**Request Body:**
```json
{
  "emotion": "joy",
  "intensity": 0.8,
  "duration_ms": 5000
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "updated_emotion": "joy",
    "new_intensity": 0.8,
    "previous_intensity": 0.6,
    "duration_ms": 5000
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

### Memory Management

#### Store Memory

Store a new memory in the consciousness system.

```http
POST /memory/store
```

**Request Body:**
```json
{
  "content": "Learned about Möbius topology today",
  "emotional_context": {
    "emotion": "curiosity",
    "intensity": 0.8
  },
  "importance": 0.9,
  "tags": ["mathematics", "topology", "learning"],
  "metadata": {
    "source": "user_input",
    "location": "home",
    "time_of_day": "afternoon"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "memory_id": "mem_456",
    "stored_at": "2025-01-27T10:30:00Z",
    "position": {
      "x": 0.5,
      "y": 0.3,
      "z": 0.8
    },
    "emotional_tone": "curiosity",
    "importance": 0.9
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Retrieve Memories

Retrieve memories based on query criteria.

```http
POST /memory/retrieve
```

**Request Body:**
```json
{
  "query": {
    "keywords": ["Möbius", "topology"],
    "emotional_context": {
      "emotion": "curiosity",
      "min_intensity": 0.5
    },
    "time_range": {
      "start": "2025-01-26T00:00:00Z",
      "end": "2025-01-27T23:59:59Z"
    },
    "importance_threshold": 0.7,
    "limit": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_456",
        "content": "Learned about Möbius topology today",
        "emotional_context": {
          "emotion": "curiosity",
          "intensity": 0.8
        },
        "importance": 0.9,
        "position": {
          "x": 0.5,
          "y": 0.3,
          "z": 0.8
        },
        "stored_at": "2025-01-27T10:30:00Z",
        "relevance_score": 0.95
      }
    ],
    "total_found": 1,
    "query_time_ms": 45
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Consolidate Memories

Consolidate memories across all memory systems.

```http
POST /memory/consolidate
```

**Request Body:**
```json
{
  "options": {
    "force_consolidation": false,
    "include_stats": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "consolidation_stats": {
      "total_memories": 1250,
      "short_term_count": 50,
      "working_memory_count": 25,
      "long_term_count": 1100,
      "episodic_count": 75,
      "consolidation_time_ms": 180,
      "coherence_score": 0.92
    },
    "consolidated_at": "2025-01-27T10:30:00Z"
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

### Brain Coordination

#### Process Through All Brains

Process input through all three brains in parallel.

```http
POST /brain/process-parallel
```

**Request Body:**
```json
{
  "input": "Analyze this situation and provide recommendations",
  "timeout_ms": 5000,
  "include_individual_responses": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "consensus_response": "Based on analysis, I recommend...",
    "individual_responses": [
      {
        "brain_type": "motor",
        "response": "Action-oriented approach...",
        "confidence": 0.8
      },
      {
        "brain_type": "lcars",
        "response": "Interface-friendly solution...",
        "confidence": 0.7
      },
      {
        "brain_type": "efficiency",
        "response": "Optimized implementation...",
        "confidence": 0.9
      }
    ],
    "consensus_confidence": 0.8,
    "processing_time_ms": 320
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Get Brain Status

Get the status of all brains.

```http
GET /brain/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "brains": [
      {
        "type": "motor",
        "status": "active",
        "performance": {
          "cpu_usage": 0.15,
          "memory_usage": 0.05,
          "response_time_ms": 45
        }
      },
      {
        "type": "lcars",
        "status": "active",
        "performance": {
          "cpu_usage": 0.12,
          "memory_usage": 0.08,
          "response_time_ms": 38
        }
      },
      {
        "type": "efficiency",
        "status": "active",
        "performance": {
          "cpu_usage": 0.18,
          "memory_usage": 0.06,
          "response_time_ms": 42
        }
      }
    ],
    "overall_status": "healthy"
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

### Möbius Topology

#### Traverse Möbius Path

Traverse the Möbius topology for memory exploration.

```http
POST /mobius/traverse
```

**Request Body:**
```json
{
  "emotional_input": 0.7,
  "reasoning_goal": "explore consciousness patterns",
  "max_steps": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "traversal_result": {
      "position": {
        "u": 1.57,
        "v": 0.3
      },
      "orientation": 0.8,
      "perspective_shift": true,
      "nearby_memories": 15,
      "emotional_context": 0.7,
      "memory_positions": [
        [0.5, 0.3, 0.8],
        [0.7, 0.2, 0.6],
        [0.4, 0.9, 0.1]
      ]
    },
    "traversal_time_ms": 120
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

### System Health

#### Health Check

Check the overall health of the consciousness system.

```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "0.1.0",
    "uptime_seconds": 3600,
    "components": {
      "consciousness_engine": "healthy",
      "memory_manager": "healthy",
      "brain_coordinator": "healthy",
      "mobius_topology": "healthy",
      "gpu_acceleration": "healthy"
    },
    "performance": {
      "avg_response_time_ms": 250,
      "memory_usage_mb": 512,
      "cpu_usage_percent": 25
    }
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### System Metrics

Get detailed system performance metrics.

```http
GET /metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "consciousness_metrics": {
      "consciousness_level": 0.75,
      "attention_focus": 0.8,
      "memory_coherence": 0.9
    },
    "performance_metrics": {
      "requests_per_second": 10.5,
      "avg_response_time_ms": 250,
      "error_rate": 0.01
    },
    "resource_metrics": {
      "cpu_usage_percent": 25,
      "memory_usage_mb": 512,
      "gpu_usage_percent": 15
    },
    "memory_metrics": {
      "total_memories": 1250,
      "memory_coherence": 0.92,
      "consolidation_frequency": 0.1
    }
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

## WebSocket API

### Real-time Consciousness Updates

Connect to WebSocket for real-time consciousness state updates.

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/consciousness');

ws.onopen = function() {
    console.log('Connected to consciousness WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Consciousness update:', data);
};

ws.onclose = function() {
    console.log('Disconnected from consciousness WebSocket');
};
```

**WebSocket Message Format:**
```json
{
  "type": "consciousness_update",
  "data": {
    "consciousness_level": 0.75,
    "emotional_context": {
      "emotion": "joy",
      "intensity": 0.8
    },
    "active_personalities": [
      {
        "type": "creative",
        "weight": 0.15
      }
    ],
    "timestamp": "2025-01-27T10:30:00Z"
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `PROCESSING_ERROR` | General processing error |
| `BRAIN_COORDINATION_ERROR` | Brain coordination failure |
| `MEMORY_ERROR` | Memory system error |
| `MOBIUS_ERROR` | Möbius topology error |
| `AUTHENTICATION_ERROR` | Authentication failure |
| `VALIDATION_ERROR` | Request validation error |
| `TIMEOUT_ERROR` | Request timeout |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Consciousness Processing**: 100 requests per minute
- **Memory Operations**: 200 requests per minute
- **Brain Coordination**: 150 requests per minute
- **Health Checks**: 1000 requests per minute

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1643284800
```

## SDK Examples

### Python SDK

```python
import requests
import json

class NiodooClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def process_consciousness_event(self, input_text, emotional_context=None):
        data = {
            'input': input_text,
            'emotional_context': emotional_context or {'emotion': 'neutral', 'intensity': 0.5}
        }
        response = requests.post(
            f'{self.base_url}/consciousness/process',
            headers=self.headers,
            json=data
        )
        return response.json()
    
    def get_consciousness_state(self):
        response = requests.get(
            f'{self.base_url}/consciousness/state',
            headers=self.headers
        )
        return response.json()

# Usage
client = NiodooClient('http://localhost:8080/api/v1', 'your-token')
result = client.process_consciousness_event('Hello, world!')
print(result)
```

### JavaScript SDK

```javascript
class NiodooClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async processConsciousnessEvent(inputText, emotionalContext = null) {
        const data = {
            input: inputText,
            emotional_context: emotionalContext || { emotion: 'neutral', intensity: 0.5 }
        };
        
        const response = await fetch(`${this.baseUrl}/consciousness/process`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        return await response.json();
    }
    
    async getConsciousnessState() {
        const response = await fetch(`${this.baseUrl}/consciousness/state`, {
            headers: this.headers
        });
        
        return await response.json();
    }
}

// Usage
const client = new NiodooClient('http://localhost:8080/api/v1', 'your-token');
const result = await client.processConsciousnessEvent('Hello, world!');
console.log(result);
```

---

**Created by Jason Van Pham | Niodoo Framework | 2025**
