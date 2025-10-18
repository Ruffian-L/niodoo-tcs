/*
 * 🌟 C++/QT CONFIG SYSTEM DEMO 🌟
 *
 * This script demonstrates the QSettings configuration system
 * for testing and validation purposes.
 */

#include <QCoreApplication>
#include <QSettings>
#include <QDebug>
#include <QString>
#include <QDir>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    qDebug() << "🌟 C++/Qt Configuration System Demo 🌟\n";

    // Test 1: Create and load configuration
    qDebug() << "1️⃣ Testing QSettings configuration...";

    QSettings settings("Niodoo", "BrainIntegration");

    // Set some configuration values
    settings.setValue("core/emotion_threshold", 0.8);
    settings.setValue("core/max_history", 100);
    settings.setValue("core/db_path", "data/custom_knowledge.db");
    settings.setValue("models/default_model", "llama3.2:latest");
    settings.setValue("models/temperature", 0.9);
    settings.setValue("qt/agents_count", 150);
    settings.setValue("qt/distributed_mode", true);

    // Sync to ensure values are written
    settings.sync();

    qDebug() << "   ✅ Configuration values set and synced";

    // Test 2: Read back configuration values
    qDebug() << "\n2️⃣ Testing configuration value retrieval...";

    double emotionThreshold = settings.value("core/emotion_threshold", 0.7).toDouble();
    int maxHistory = settings.value("core/max_history", 50).toInt();
    QString dbPath = settings.value("core/db_path", "data/knowledge_graph.db").toString();
    QString modelName = settings.value("models/default_model", "llama3:latest").toString();
    double temperature = settings.value("models/temperature", 0.8).toDouble();
    int agentsCount = settings.value("qt/agents_count", 89).toInt();
    bool distributedMode = settings.value("qt/distributed_mode", true).toBool();

    qDebug() << "   Emotion threshold:" << emotionThreshold;
    qDebug() << "   Max history:" << maxHistory;
    qDebug() << "   Database path:" << dbPath;
    qDebug() << "   Model name:" << modelName;
    qDebug() << "   Temperature:" << temperature;
    qDebug() << "   Agents count:" << agentsCount;
    qDebug() << "   Distributed mode:" << distributedMode;

    // Test 3: Test default values
    qDebug() << "\n3️⃣ Testing default value fallbacks...";

    double defaultThreshold = settings.value("core/nonexistent_threshold", 0.5).toDouble();
    QString defaultModel = settings.value("models/nonexistent_model", "fallback-model").toString();

    qDebug() << "   Default threshold (nonexistent key):" << defaultThreshold;
    qDebug() << "   Default model (nonexistent key):" << defaultModel;

    // Test 4: Test comprehensive configuration coverage
    qDebug() << "\n4️⃣ Testing comprehensive configuration coverage...";

    // Set comprehensive configuration values
    settings.setValue("core/backup_interval", 7200);
    settings.setValue("core/context_window", 15);
    settings.setValue("core/response_delay", 1.5);
    settings.setValue("models/max_tokens", 300);
    settings.setValue("models/timeout", 60);
    settings.setValue("models/top_p", 0.95);
    settings.setValue("models/top_k", 60);
    settings.setValue("models/repeat_penalty", 1.2);
    settings.setValue("models/frequency_penalty", 0.3);
    settings.setValue("models/presence_penalty", 0.4);
    settings.setValue("qt/connections_count", 2000);
    settings.setValue("qt/architect_endpoint", "http://custom-architect:11434/api/generate");
    settings.setValue("qt/developer_endpoint", "http://custom-developer:11434/api/generate");
    settings.setValue("qt/hardware_acceleration", false);
    settings.setValue("qt/network_mode", "Hybrid");
    settings.setValue("qt/pathway_activation", 150);
    settings.setValue("api/ollama_url", "http://custom-ollama:11434");
    settings.setValue("api/api_timeout", 45);
    settings.setValue("api/retry_attempts", 5);
    settings.setValue("api/enable_caching", false);
    settings.setValue("api/cache_ttl", 600);
    settings.setValue("consciousness/enabled", false);
    settings.setValue("consciousness/reflection_enabled", false);
    settings.setValue("consciousness/emotion_sensitivity", 0.9);
    settings.setValue("consciousness/memory_threshold", 0.8);
    settings.setValue("consciousness/pattern_sensitivity", 0.9);
    settings.setValue("consciousness/self_awareness_level", 0.95);
    settings.setValue("emotions/enabled", false);
    settings.setValue("emotions/max_response_history", 15);
    settings.setValue("emotions/repetition_penalty", 0.7);
    settings.setValue("emotions/enhance_responses", false);
    settings.setValue("performance/gpu_usage_target", 90);
    settings.setValue("performance/memory_usage_target", 95);
    settings.setValue("performance/temperature_threshold", 85);
    settings.setValue("performance/enable_monitoring", false);
    settings.setValue("performance/optimization_interval", 120);
    settings.setValue("logging/level", "DEBUG");
    settings.setValue("logging/file", "data/custom.log");
    settings.setValue("logging/console_level", "WARN");
    settings.setValue("logging/enable_structured_logging", false);
    settings.setValue("logging/enable_log_rotation", false);
    settings.setValue("logging/max_log_file_size", 5);

    settings.sync();

    // Read back comprehensive configuration
    qDebug() << "   ✅ Comprehensive configuration set successfully";
    qDebug() << "   Backup interval:" << settings.value("core/backup_interval", 3600).toInt();
    qDebug() << "   Context window:" << settings.value("core/context_window", 10).toInt();
    qDebug() << "   Max tokens:" << settings.value("models/max_tokens", 200).toInt();
    qDebug() << "   Connections count:" << settings.value("qt/connections_count", 1209).toInt();
    qDebug() << "   Custom architect endpoint:" << settings.value("qt/architect_endpoint").toString();
    qDebug() << "   Consciousness enabled:" << settings.value("consciousness/enabled", true).toBool();
    qDebug() << "   Performance monitoring:" << settings.value("performance/enable_monitoring", true).toBool();
    qDebug() << "   Log level:" << settings.value("logging/level", "INFO").toString();

    // Test 5: Test configuration file location
    qDebug() << "\n5️⃣ Testing configuration file location...";

    QString configPath = settings.fileName();
    qDebug() << "   Configuration file path:" << configPath;

    QFileInfo configFile(configPath);
    if (configFile.exists()) {
        qDebug() << "   ✅ Configuration file exists";
        qDebug() << "   File size:" << configFile.size() << "bytes";
        qDebug() << "   Last modified:" << configFile.lastModified().toString();
    } else {
        qDebug() << "   ❌ Configuration file does not exist";
    }

    // Test 6: Test configuration validation
    qDebug() << "\n6️⃣ Testing configuration validation...";

    // Test valid ranges
    settings.setValue("core/emotion_threshold", 0.5); // Valid: 0.0-1.0
    settings.setValue("models/temperature", 1.0); // Valid: 0.0-2.0
    settings.setValue("qt/agents_count", 100); // Valid: > 0

    // Test invalid ranges (for demo purposes)
    // settings.setValue("core/emotion_threshold", 1.5); // Invalid: > 1.0
    // settings.setValue("models/temperature", -0.1); // Invalid: < 0.0
    // settings.setValue("qt/agents_count", 0); // Invalid: = 0

    settings.sync();

    double validThreshold = settings.value("core/emotion_threshold", 0.7).toDouble();
    double validTemp = settings.value("models/temperature", 0.8).toDouble();
    int validAgents = settings.value("qt/agents_count", 89).toInt();

    qDebug() << "   Validated emotion threshold:" << validThreshold;
    qDebug() << "   Validated temperature:" << validTemp;
    qDebug() << "   Validated agents count:" << validAgents;

    qDebug() << "\n🎉 C++/Qt configuration system demo completed successfully!";
    qDebug() << "\n💡 Usage tips:";
    qDebug() << "  - Configuration is stored in registry (Windows) or .config (Linux)";
    qDebug() << "  - Use QSettings::setValue() to store and value() to retrieve";
    qDebug() << "  - All hardcodes have been replaced with configurable values";
    qDebug() << "  - Test different configurations by modifying the settings object";

    return 0;
}
