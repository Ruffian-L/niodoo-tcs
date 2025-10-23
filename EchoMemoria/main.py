#!/usr/bin/env python3
"""
🌟 REAL AI INFERENCE - C++/RUST/QML SYNTHESIS 🌟

EchoMemoria v3.0 - NO MORE PYTHON BULLSHIT
Pure C++/Rust/QML consciousness synthesis with REAL AI inference
"""

import subprocess
import sys
import json
import time
import argparse
import os
import asyncio
from core.integrated_consciousness import IntegratedConsciousness

def parse_args():
    parser = argparse.ArgumentParser(description='Niodoo Integrated Consciousness - Async Pipeline')
    parser.add_argument('--prompt', type=str, help='Test prompt for end-to-end pipeline')
    parser.add_argument('--config', type=str, default='config/settings.json', help='Config path')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')

    # Core settings
    core_group = parser.add_argument_group('Core Settings')
    core_group.add_argument('--max-history', type=int,
                           help='Maximum conversation history (overrides config)')
    core_group.add_argument('--emotion-threshold', type=float,
                           help='Emotion detection threshold (overrides config)')
    core_group.add_argument('--db-path', type=str,
                           help='Database path for knowledge storage (overrides config)')
    core_group.add_argument('--backup-interval', type=int,
                           help='Backup interval in seconds (overrides config)')
    core_group.add_argument('--context-window', type=int,
                           help='Context window size for processing (overrides config)')
    core_group.add_argument('--response-delay', type=float,
                           help='Response delay in seconds (overrides config)')

    # Model settings
    model_group = parser.add_argument_group('AI Model Settings')
    model_group.add_argument('--model', type=str,
                            help='Ollama model (overrides config)')
    model_group.add_argument('--backup-model', type=str,
                            help='Backup model (overrides config)')
    model_group.add_argument('--temperature', type=float,
                            help='Ollama temperature (overrides config)')
    model_group.add_argument('--max-tokens', type=int,
                            help='Maximum tokens per response (overrides config)')
    model_group.add_argument('--timeout', type=int,
                            help='Timeout for model requests (overrides config)')
    model_group.add_argument('--top-p', type=float,
                            help='Top-p sampling parameter (overrides config)')
    model_group.add_argument('--top-k', type=int,
                            help='Top-k sampling parameter (overrides config)')
    model_group.add_argument('--repeat-penalty', type=float,
                            help='Repeat penalty (overrides config)')
    model_group.add_argument('--frequency-penalty', type=float,
                            help='Frequency penalty (overrides config)')
    model_group.add_argument('--presence-penalty', type=float,
                            help='Presence penalty (overrides config)')

    # RAG settings
    rag_group = parser.add_argument_group('RAG Settings')
    rag_group.add_argument('--rag-enabled', action='store_true',
                          help='Enable RAG functionality (overrides config)')
    rag_group.add_argument('--rag-chunk-size', type=int,
                          help='Chunk size for text splitting (overrides config)')
    rag_group.add_argument('--rag-similarity-threshold', type=float,
                          help='Similarity threshold for retrieval (overrides config)')
    rag_group.add_argument('--rag-context-limit', type=int,
                          help='Maximum context length (overrides config)')
    rag_group.add_argument('--rag-inspiration-mode', action='store_true',
                          help='Enable inspiration mode (overrides config)')
    rag_group.add_argument('--rag-ingestion-batch-size', type=int,
                          help='Ingestion batch size (overrides config)')

    # Training settings
    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--learning-rate', type=float,
                            help='Learning rate for training (overrides config)')
    train_group.add_argument('--epochs', type=int,
                            help='Number of training epochs (overrides config)')
    train_group.add_argument('--hidden-dim', type=int,
                            help='Hidden layer dimensions (overrides config)')
    train_group.add_argument('--input-dim', type=int,
                            help='Input dimensions (overrides config)')
    train_group.add_argument('--output-dim', type=int,
                            help='Output dimensions (overrides config)')

    # Emotion settings
    emotion_group = parser.add_argument_group('Emotion Settings')
    emotion_group.add_argument('--emotions-enabled', action='store_true',
                              help='Enable emotion processing (overrides config)')
    emotion_group.add_argument('--response-types', nargs='+',
                              help='Supported response types (overrides config)')
    emotion_group.add_argument('--max-response-history', type=int,
                              help='Maximum response history (overrides config)')
    emotion_group.add_argument('--repetition-penalty', type=float,
                              help='Repetition penalty (overrides config)')
    emotion_group.add_argument('--enhance-responses', action='store_true',
                              help='Enable emotion enhancement (overrides config)')

    # Consciousness settings
    consciousness_group = parser.add_argument_group('Consciousness Settings')
    consciousness_group.add_argument('--consciousness-enabled', action='store_true',
                                    help='Enable consciousness processing (overrides config)')
    consciousness_group.add_argument('--reflection-enabled', action='store_true',
                                    help='Enable reflection processing (overrides config)')
    consciousness_group.add_argument('--emotion-sensitivity', type=float,
                                    help='Emotion sensitivity level (overrides config)')
    consciousness_group.add_argument('--memory-threshold', type=float,
                                    help='Memory formation threshold (overrides config)')
    consciousness_group.add_argument('--pattern-sensitivity', type=float,
                                    help='Pattern recognition sensitivity (overrides config)')
    consciousness_group.add_argument('--self-awareness-level', type=float,
                                    help='Self-awareness level (overrides config)')

    # Performance settings
    perf_group = parser.add_argument_group('Performance Settings')
    perf_group.add_argument('--diversity-temperature-boost', type=float,
                           help='Diversity temperature boost (overrides config)')
    perf_group.add_argument('--max-diversity-temperature', type=float,
                           help='Maximum diversity temperature (overrides config)')
    perf_group.add_argument('--response-similarity-threshold', type=float,
                           help='Response similarity threshold (overrides config)')
    perf_group.add_argument('--fallback-mode-enabled', action='store_true',
                           help='Enable fallback mode (overrides config)')

    # Logging settings
    log_group = parser.add_argument_group('Logging Settings')
    log_group.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
                          help='Log level (overrides config)')
    log_group.add_argument('--log-file', type=str,
                          help='Log file path (overrides config)')
    log_group.add_argument('--console-log-level', type=str, choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
                          help='Console log level (overrides config)')

    # API settings
    api_group = parser.add_argument_group('API Settings')
    api_group.add_argument('--ollama-url', type=str,
                          help='Ollama API URL (overrides config)')
    api_group.add_argument('--api-timeout', type=int,
                          help='API timeout in seconds (overrides config)')
    api_group.add_argument('--retry-attempts', type=int,
                          help='Retry attempts for API calls (overrides config)')
    api_group.add_argument('--enable-caching', action='store_true',
                          help='Enable API caching (overrides config)')
    api_group.add_argument('--cache-ttl', type=int,
                          help='Cache TTL in seconds (overrides config)')

    return parser.parse_args()

async def run_pipeline_test(consciousness: IntegratedConsciousness, prompt: str):
    """Run end-to-end async pipeline test with JSON output"""
    print(f"🚀 Testing async pipeline with prompt: {prompt[:50]}...")
    result = await consciousness.process_interaction_async(prompt)
    
    print("\n📊 PIPELINE RESULTS:")
    print(f"Response: {result['response']}")
    print(f"Latency: {result.get('total_latency_ms', 'N/A'):.0f}ms")
    print(f"Memories: {result['relevant_memories']}")
    print(f"Confidence: {result['mobius_confidence']:.2f}")
    
    # Output full JSON
    print("\n📄 FULL JSON OUTPUT:")
    print(result['json_output'])
    
    return result

def main():
    args = parse_args()
    
    # Initialize consciousness (existing logic)
    consciousness = IntegratedConsciousness(
        config_path=args.config,
        enable_visualization=not args.no_viz
    )
    
    if args.prompt:
        # Test mode: run single pipeline invocation
        asyncio.run(run_pipeline_test(consciousness, args.prompt))
    else:
        # Existing interactive loop (updated to async)
        conversation_history = []
        print("🧠 Niodoo Async Consciousness Active! Use --prompt for testing or type 'quit'")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'quit':
                    break
                
                # Run async pipeline
                result = loop.run_until_complete(consciousness.process_interaction_async(user_input))
                
                response = result['response']
                print(f"Niodo: {response}")
                print(f"   Latency: {result.get('total_latency_ms', 'N/A'):.0f}ms")
                
                # Brief metrics
                if result['relevant_memories'] > 0:
                    print(f"   🧠 Used {result['relevant_memories']} memories, Confidence: {result['mobius_confidence']:.2f}")
                
                # Update history (existing)
                conversation_history.append(f"You: {user_input}")
                conversation_history.append(f"Niodo: {response}")
                
                # Limit history (existing)
                max_history = 50  # From config
                if len(conversation_history) > max_history:
                    conversation_history = conversation_history[-max_history:]
        finally:
            loop.close()
            consciousness.shutdown()

if __name__ == "__main__":
    main()
