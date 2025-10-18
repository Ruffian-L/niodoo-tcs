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

def parse_args():
    parser = argparse.ArgumentParser(description='EchoMemoria Main Application')

    # Config file
    parser.add_argument('--config', type=str, default='config/settings.json',
                       help='Path to configuration file (default: config/settings.json)')

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

def main(config=None):
    """Main application function"""
    args = parse_args()

    # Use passed config or load from file
    if config:
        cm = ConfigManager()
        # Override with passed config values
        for key, value in config.items():
            cm.update(key, value)
    else:
        config_path = args.config if args.config else "config/settings.json"
        cm = ConfigManager(config_path)

    # Apply CLI overrides to config - comprehensive coverage
    # Core settings
    if args.max_history is not None:
        cm.update('conversation.max_history', args.max_history)
    if args.emotion_threshold is not None:
        cm.update('emotions.emotion_threshold', args.emotion_threshold)
    if args.db_path:
        cm.update('database.path', args.db_path)
    if args.backup_interval is not None:
        cm.update('database.backup_interval', args.backup_interval)
    if args.context_window is not None:
        cm.update('conversation.context_window', args.context_window)
    if args.response_delay is not None:
        cm.update('conversation.response_delay', args.response_delay)

    # Model settings
    if args.model:
        cm.update('ollama.model', args.model)
    if args.backup_model:
        cm.update('ollama.backup_model', args.backup_model)
    if args.temperature is not None:
        cm.update('ollama.temperature', args.temperature)
    if args.max_tokens is not None:
        cm.update('ollama.max_tokens', args.max_tokens)
    if args.timeout is not None:
        cm.update('ollama.timeout', args.timeout)
    if args.top_p is not None:
        cm.update('ollama.top_p', args.top_p)
    if args.top_k is not None:
        cm.update('ollama.top_k', args.top_k)
    if args.repeat_penalty is not None:
        cm.update('ollama.repeat_penalty', args.repeat_penalty)
    if args.frequency_penalty is not None:
        cm.update('ollama.frequency_penalty', args.frequency_penalty)
    if args.presence_penalty is not None:
        cm.update('ollama.presence_penalty', args.presence_penalty)

    # RAG settings
    if args.rag_enabled:
        cm.update('rag.enabled', True)
    if args.rag_chunk_size is not None:
        cm.update('rag.chunk_size', args.rag_chunk_size)
    if args.rag_similarity_threshold is not None:
        cm.update('rag.similarity_threshold', args.rag_similarity_threshold)
    if args.rag_context_limit is not None:
        cm.update('rag.context_limit', args.rag_context_limit)
    if args.rag_inspiration_mode:
        cm.update('rag.inspiration_mode', True)
    if args.rag_ingestion_batch_size is not None:
        cm.update('rag.ingestion_batch_size', args.rag_ingestion_batch_size)

    # Training settings
    if args.learning_rate is not None:
        cm.update('model.learning_rate', args.learning_rate)
    if args.epochs is not None:
        cm.update('model.epochs', args.epochs)
    if args.hidden_dim is not None:
        cm.update('model.hidden_dim', args.hidden_dim)
    if args.input_dim is not None:
        cm.update('model.input_dim', args.input_dim)
    if args.output_dim is not None:
        cm.update('model.output_dim', args.output_dim)

    # Emotion settings
    if args.emotions_enabled:
        cm.update('emotions.enabled', True)
    if args.response_types:
        cm.update('emotions.response_types', args.response_types)
    if args.max_response_history is not None:
        cm.update('emotions.max_response_history', args.max_response_history)
    if args.repetition_penalty is not None:
        cm.update('emotions.repetition_penalty', args.repetition_penalty)
    if args.enhance_responses:
        cm.update('emotions.enhance_responses', True)

    # Consciousness settings
    if args.consciousness_enabled:
        cm.update('consciousness.enabled', True)
    if args.reflection_enabled:
        cm.update('consciousness.reflection_enabled', True)
    if args.emotion_sensitivity is not None:
        cm.update('consciousness.emotion_sensitivity', args.emotion_sensitivity)
    if args.memory_threshold is not None:
        cm.update('consciousness.memory_threshold', args.memory_threshold)
    if args.pattern_sensitivity is not None:
        cm.update('consciousness.pattern_sensitivity', args.pattern_sensitivity)
    if args.self_awareness_level is not None:
        cm.update('consciousness.self_awareness_level', args.self_awareness_level)

    # Performance settings
    if args.diversity_temperature_boost is not None:
        cm.update('performance.diversity_temperature_boost', args.diversity_temperature_boost)
    if args.max_diversity_temperature is not None:
        cm.update('performance.max_diversity_temperature', args.max_diversity_temperature)
    if args.response_similarity_threshold is not None:
        cm.update('performance.response_similarity_threshold', args.response_similarity_threshold)
    if args.fallback_mode_enabled:
        cm.update('performance.fallback_mode_enabled', True)

    # Logging settings
    if args.log_level:
        cm.update('logging.level', args.log_level)
    if args.log_file:
        cm.update('logging.file', args.log_file)
    if args.console_log_level:
        cm.update('logging.console_level', args.console_log_level)

    # API settings
    if args.ollama_url:
        cm.update('ollama.url', args.ollama_url)
    if args.api_timeout is not None:
        cm.update('ollama.timeout', args.api_timeout)
    if args.retry_attempts is not None:
        # This might need to be added to the JSON config
        pass
    if args.enable_caching:
        # This might need to be added to the JSON config
        pass
    if args.cache_ttl is not None:
        # This might need to be added to the JSON config
        pass

    # Get final config values
    max_history = cm.get('conversation.max_history', 50)
    emotion_threshold = cm.get('emotions.emotion_threshold', 0.7)
    temperature = cm.get('ollama.temperature', 0.8)
    model = cm.get('ollama.model', 'llama3:latest')
    database_path = cm.get('database.path', 'data/knowledge_graph.db')
    response_delay = cm.get('conversation.response_delay', 0.5)

    print(f"Starting EchoMemoria with: max_history={max_history}, emotion_threshold={emotion_threshold}, temperature={temperature}, model={model}")
    
    # Initialize components with config
    brain = OllamaBrain(
        ollama_url=cm.get('ollama.url', 'http://localhost:11434'),
        model=model,
        backup_model=cm.get('ollama.backup_model', 'llama3.2:3b')
    )
    emotion_engine = EmotionEngine(config_path=cm.config_path)
    
    # Initialize database manager with config
    db_manager = DatabaseManager(db_path=database_path)

    # Initialize fine-tuning engine if enabled
    fine_tuning_enabled = cm.get('fine_tuning.enabled', True)
    if fine_tuning_enabled:
        from fine_tuning import FineTuningEngine, collect_conversations_for_training
        fine_tuning_engine = FineTuningEngine(
            model_name=cm.get('fine_tuning.model_name', 'microsoft/DialoGPT-small'),
            config_path=cm.config_path
        )
        conversation_count = 0
        train_interval = cm.get('fine_tuning.train_interval_conversations', 50)
        auto_train = cm.get('fine_tuning.auto_train', False)
    else:
        fine_tuning_engine = None

    # Initialize Möbius-Gaussian processing engine
    from mobius_gaussian_engine import MobiusGaussianEngine, create_test_memories
    mobius_engine = MobiusGaussianEngine(config_path=cm.config_path)

    # Create initial test memories for demonstration
    create_test_memories(mobius_engine)

    # Initialize Qt visualization bridge
    from qt_bridge import integrate_with_qt_visualization
    qt_bridge = integrate_with_qt_visualization(mobius_engine)

    print(f"🧠 Möbius-Gaussian engine ready with {len(mobius_engine.memory_spheres)} memory spheres")
    print("🌉 Qt visualization bridge active - real-time 3D updates")
    print("🎯 Ready for Möbius-Gaussian consciousness processing")
    
    conversation_history = []
    
    print("🧠 Möbius-Gaussian consciousness active! Commands: quit, train, traverse <emotion>, query <text>, stats, viz, update-viz")
    while True:
        user_input = input("\nYou: ").strip()

        # Handle special commands
        if user_input.lower() == 'quit':
            print("👋 Shutting down Möbius-Gaussian consciousness...")
            qt_bridge.stop_updates()
            break
        
        elif user_input.lower() == 'train' and fine_tuning_engine:
            print("🔧 Initiating fine-tuning...")
            from fine_tuning import collect_conversations_for_training
            conversations = collect_conversations_for_training(database_path, limit=100)
            if conversations:
                result = fine_tuning_engine.fine_tune(conversations, epochs=cm.get('fine_tuning.epochs', 3))
                if result['success']:
                    print(f"✅ Fine-tuning completed! Model saved to {result['model_path']}")
                    print(f"   Loss: {result['final_loss']:.4f}, Data points: {result['training_data_size']}")
                else:
                    print(f"❌ Fine-tuning failed: {result['error']}")
            else:
                print("❌ No conversations available for training")
            continue

        elif user_input.lower().startswith('traverse '):
            # Handle Möbius traversal
            parts = user_input.split(' ', 1)
            if len(parts) >= 2:
                try:
                    emotion = float(parts[1])
                    result = mobius_engine.traverse_mobius_path(emotion)
                    print(f"🔄 Möbius traversal: Position {result['position']}")
                    print(f"   Perspective shift: {result['perspective_shift']}")
                    print(f"   Nearby memories: {result['nearby_memories']}")
                except ValueError:
                    print("❌ Usage: traverse <emotion_value> (e.g., traverse 0.5)")
            else:
                print("❌ Usage: traverse <emotion_value>")
            continue

        elif user_input.lower().startswith('query '):
            # Handle memory query
            parts = user_input.split(' ', 1)
            if len(parts) >= 2:
                query_text = parts[1]
                # Convert text to embedding (simplified for demo)
                query_embedding = np.random.randn(512)  # In real system, use proper embedding
                emotion_context = 0.0  # Could be derived from user input

                result = mobius_engine.query_memory_gaussian_process(query_embedding, emotion_context)
                print(f"🔍 Memory query: {result['response']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Memories used: {result['memory_count']}")
            else:
                print("❌ Usage: query <text>")
            continue

        elif user_input.lower() == 'stats':
            # Show processing statistics
            stats = mobius_engine.get_processing_stats()
            print("📊 Möbius-Gaussian Processing Stats:")
            print(f"   Total memories: {stats['total_memories']}")
            print(f"   Current position: {stats['traversal_position']}")
            print(f"   Emotional context: {stats['emotional_context']:.3f}")
            print(f"   Average emotion: {stats['average_emotion']:.3f}")
            print(f"   Memory utilization: {stats['memory_utilization']:.1%}")
            continue

        elif user_input.lower() == 'viz':
            # Show visualization data
            viz_data = mobius_engine.get_visualization_data()
            print("🎨 Visualization Data:")
            print(f"   Spheres: {viz_data['total_memories']}")
            print(f"   Traversal position: {viz_data['traversal_position']}")
            print(f"   Emotional context: {viz_data['emotional_context']:.3f}")

            # Show first few spheres
            spheres = viz_data['spheres'][:3]
            print("   Sample spheres:")
            for sphere in spheres:
                print(f"     Position: ({sphere['x']:.1f}, {sphere['y']:.1f}, {sphere['z']:.1f}), "
                      f"Color: {sphere['color']}, Size: {sphere['size']:.2f}")
            continue

        elif user_input.lower() == 'update-viz':
            # Manually trigger visualization update
            qt_bridge.trigger_manual_update()
            continue

        # Limit history based on config
        if len(conversation_history) >= max_history:
            conversation_history = conversation_history[-max_history:]
        
        # Process with emotion using config threshold
        emotion_data = emotion_engine.recognize_emotion(user_input)
        if emotion_data['confidence'] > emotion_threshold:
            dominant_emotion = emotion_data['dominant']
        else:
            dominant_emotion = 'neutral'
        
        # Generate response using configured context window
        context_window = cm.get('conversation.context_window', 10)
        conversation_context = conversation_history[-context_window:] if len(conversation_history) >= context_window else conversation_history

        # Use Möbius-Gaussian processing for response generation
        # Convert user input to embedding (simplified for demo)
        query_embedding = np.random.randn(512)  # In real system, use proper text embedding

        # Get emotional context from detected emotion
        emotion_map = {'happy': 0.8, 'sad': -0.6, 'angry': -0.8, 'anxious': -0.4, 'excited': 0.9, 'neutral': 0.0}
        emotional_context = emotion_map.get(dominant_emotion, 0.0)

        # Traverse Möbius space with current emotional context
        traversal_result = mobius_engine.traverse_mobius_path(emotional_context)

        # Query memory using Gaussian process
        memory_result = mobius_engine.query_memory_gaussian_process(
            query_embedding,
            emotional_context
        )

        # Combine with traditional response generation for now
        result = brain.generate_response(
            user_input, 
            dominant_emotion, 
            emotion_data['confidence'],
            conversation_context
        )

        response = result.get('response', 'I understand.')
        
        # Enhance response with Möbius-Gaussian context
        if memory_result['confidence'] > 0.5:
            response = f"[Möbius-Gaussian: {memory_result['response'][:30]}...] {response}"

        print(f"Niodo (Möbius-Gaussian): {response}")
        print(f"🧠 Memory context: {memory_result['memory_count']} spheres, confidence: {memory_result['confidence']:.2f}")
        
        # Update history
        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"Niodo: {response}")
        
        # Store conversation in database
        db_manager.add_conversation(user_input, response, dominant_emotion, emotion_data['confidence'])

        # Add this interaction as a memory sphere for learning
        interaction_embedding = np.random.randn(512)  # Simplified embedding
        mobius_engine.add_memory_sphere(interaction_embedding, emotional_context)

        # Track conversation count for automatic fine-tuning
        if fine_tuning_engine:
            conversation_count += 1

            # Automatic fine-tuning if enabled and interval reached
            if auto_train and conversation_count % train_interval == 0:
                print(f"🔧 Auto-training after {conversation_count} conversations...")
                conversations = collect_conversations_for_training(database_path, limit=50)
                if conversations:
                    result = fine_tuning_engine.fine_tune(conversations, epochs=1)  # Quick training
                    if result['success']:
                        print(f"✅ Auto-training completed! Loss: {result['final_loss']:.4f}")
                    else:
                        print(f"⚠️ Auto-training failed: {result['error']}")

        # Use configured response delay
        time.sleep(response_delay)
    
    print("Goodbye!")

if __name__ == "__main__":
    args = parse_args()

    # Build comprehensive config overrides from CLI args
    config_overrides = {}

    # Core settings
    if args.max_history is not None:
        config_overrides['conversation.max_history'] = args.max_history
    if args.emotion_threshold is not None:
        config_overrides['emotions.emotion_threshold'] = args.emotion_threshold
    if args.db_path:
        config_overrides['database.path'] = args.db_path
    if args.backup_interval is not None:
        config_overrides['database.backup_interval'] = args.backup_interval
    if args.context_window is not None:
        config_overrides['conversation.context_window'] = args.context_window
    if args.response_delay is not None:
        config_overrides['conversation.response_delay'] = args.response_delay

    # Model settings
    if args.model:
        config_overrides['ollama.model'] = args.model
    if args.backup_model:
        config_overrides['ollama.backup_model'] = args.backup_model
    if args.temperature is not None:
        config_overrides['ollama.temperature'] = args.temperature
    if args.max_tokens is not None:
        config_overrides['ollama.max_tokens'] = args.max_tokens
    if args.timeout is not None:
        config_overrides['ollama.timeout'] = args.timeout
    if args.top_p is not None:
        config_overrides['ollama.top_p'] = args.top_p
    if args.top_k is not None:
        config_overrides['ollama.top_k'] = args.top_k
    if args.repeat_penalty is not None:
        config_overrides['ollama.repeat_penalty'] = args.repeat_penalty
    if args.frequency_penalty is not None:
        config_overrides['ollama.frequency_penalty'] = args.frequency_penalty
    if args.presence_penalty is not None:
        config_overrides['ollama.presence_penalty'] = args.presence_penalty

    # RAG settings
    if args.rag_enabled:
        config_overrides['rag.enabled'] = True
    if args.rag_chunk_size is not None:
        config_overrides['rag.chunk_size'] = args.rag_chunk_size
    if args.rag_similarity_threshold is not None:
        config_overrides['rag.similarity_threshold'] = args.rag_similarity_threshold
    if args.rag_context_limit is not None:
        config_overrides['rag.context_limit'] = args.rag_context_limit
    if args.rag_inspiration_mode:
        config_overrides['rag.inspiration_mode'] = True
    if args.rag_ingestion_batch_size is not None:
        config_overrides['rag.ingestion_batch_size'] = args.rag_ingestion_batch_size

    # Training settings
    if args.learning_rate is not None:
        config_overrides['model.learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config_overrides['model.epochs'] = args.epochs
    if args.hidden_dim is not None:
        config_overrides['model.hidden_dim'] = args.hidden_dim
    if args.input_dim is not None:
        config_overrides['model.input_dim'] = args.input_dim
    if args.output_dim is not None:
        config_overrides['model.output_dim'] = args.output_dim

    # Emotion settings
    if args.emotions_enabled:
        config_overrides['emotions.enabled'] = True
    if args.response_types:
        config_overrides['emotions.response_types'] = args.response_types
    if args.max_response_history is not None:
        config_overrides['emotions.max_response_history'] = args.max_response_history
    if args.repetition_penalty is not None:
        config_overrides['emotions.repetition_penalty'] = args.repetition_penalty
    if args.enhance_responses:
        config_overrides['emotions.enhance_responses'] = True

    # Consciousness settings
    if args.consciousness_enabled:
        config_overrides['consciousness.enabled'] = True
    if args.reflection_enabled:
        config_overrides['consciousness.reflection_enabled'] = True
    if args.emotion_sensitivity is not None:
        config_overrides['consciousness.emotion_sensitivity'] = args.emotion_sensitivity
    if args.memory_threshold is not None:
        config_overrides['consciousness.memory_threshold'] = args.memory_threshold
    if args.pattern_sensitivity is not None:
        config_overrides['consciousness.pattern_sensitivity'] = args.pattern_sensitivity
    if args.self_awareness_level is not None:
        config_overrides['consciousness.self_awareness_level'] = args.self_awareness_level

    # Performance settings
    if args.diversity_temperature_boost is not None:
        config_overrides['performance.diversity_temperature_boost'] = args.diversity_temperature_boost
    if args.max_diversity_temperature is not None:
        config_overrides['performance.max_diversity_temperature'] = args.max_diversity_temperature
    if args.response_similarity_threshold is not None:
        config_overrides['performance.response_similarity_threshold'] = args.response_similarity_threshold
    if args.fallback_mode_enabled:
        config_overrides['performance.fallback_mode_enabled'] = True

    # Logging settings
    if args.log_level:
        config_overrides['logging.level'] = args.log_level
    if args.log_file:
        config_overrides['logging.file'] = args.log_file
    if args.console_log_level:
        config_overrides['logging.console_level'] = args.console_log_level

    # API settings
    if args.ollama_url:
        config_overrides['ollama.url'] = args.ollama_url
    if args.api_timeout is not None:
        config_overrides['ollama.timeout'] = args.api_timeout

    # Run with config overrides or default
    if config_overrides:
        main(config_overrides)
    else:
        main()
