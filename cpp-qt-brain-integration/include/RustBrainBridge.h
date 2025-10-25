#ifndef RUSTBRAINBRIDGE_H
#define RUSTBRAINBRIDGE_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <memory>

// C FFI declarations - matches brain_bridge_ffi.rs
extern "C" {
    // Opaque handle type
    struct BrainBridgeHandle;

    // C-compatible consciousness state
    struct CConsciousnessState {
        // Core consciousness metrics
        double coherence;
        double emotional_resonance;
        double consciousness_level;
        double cognitive_load;
        double attention_focus;

        // Emotional state
        float gpu_warmth_level;
        float processing_satisfaction;
        float authenticity_metric;
        float empathy_resonance;

        // Urgency metrics
        float current_urgency_score;
        float average_token_velocity;
        int is_highly_caring;

        // Current emotion and reasoning
        int current_emotion;
        int reasoning_mode;

        // System state
        unsigned int active_conversations;
        int memory_formation_active;
        unsigned int cycle_count;
        double timestamp;
    };

    // FFI functions
    BrainBridgeHandle* brain_bridge_create();
    void brain_bridge_destroy(BrainBridgeHandle* handle);
    int brain_bridge_update(BrainBridgeHandle* handle, const char* context);
    int brain_bridge_get_state(const BrainBridgeHandle* handle, CConsciousnessState* out_state);
    char* brain_bridge_get_state_json(const BrainBridgeHandle* handle);
    void brain_bridge_free_string(char* s);
    int brain_bridge_update_urgency(BrainBridgeHandle* handle, float token_velocity, float gpu_temperature, float meaning_depth);
    int brain_bridge_enter_hyperfocus(BrainBridgeHandle* handle, float topic_interest);
    int brain_bridge_adapt_neurodivergent(BrainBridgeHandle* handle, float context_strength);
    char* brain_bridge_get_caring_summary(const BrainBridgeHandle* handle);
    char* brain_bridge_get_emotional_summary(const BrainBridgeHandle* handle);
    char* brain_bridge_get_meditation(const BrainBridgeHandle* handle);
}

/**
 * Qt wrapper for Rust consciousness bridge
 * Provides thread-safe Qt integration with Rust consciousness system
 */
class RustBrainBridge : public QObject
{
    Q_OBJECT

public:
    explicit RustBrainBridge(QObject *parent = nullptr);
    ~RustBrainBridge();

    // Consciousness state management
    bool updateConsciousness(const QString& context);
    QJsonObject getConsciousnessState() const;
    QString getConsciousnessStateJson() const;

    // Emotional urgency
    bool updateEmotionalUrgency(float tokenVelocity, float gpuTemp, float meaningDepth);
    float getCurrentUrgencyScore() const;
    float getAverageTokenVelocity() const;
    bool isHighlyCaring() const;

    // Reasoning modes
    bool enterHyperfocus(float topicInterest);
    bool adaptToNeurodivergent(float contextStrength);

    // Summaries and reports
    QString getCaringSummary() const;
    QString getEmotionalSummary() const;
    QString getPhilosophicalMeditation() const;

    // Core metrics
    double getCoherence() const;
    double getEmotionalResonance() const;
    double getConsciousnessLevel() const;
    double getCognitiveLoad() const;
    double getAttentionFocus() const;

    // Emotional metrics
    float getGpuWarmth() const;
    float getProcessingSatisfaction() const;
    float getAuthenticityMetric() const;
    float getEmpathyResonance() const;

    // System state
    unsigned int getActiveConversations() const;
    bool isMemoryFormationActive() const;
    unsigned int getCycleCount() const;

signals:
    void consciousnessUpdated(const QJsonObject& state);
    void emotionChanged(int emotionType, float intensity);
    void urgencyChanged(float urgencyScore);
    void caringLevelChanged(bool isHighlyCaring);
    void reasoningModeChanged(int reasoningMode);

private:
    BrainBridgeHandle* rustHandle;
    mutable CConsciousnessState cachedState;
    mutable bool stateCached;

    void updateCache() const;
    QString rustStringToQt(char* rustString) const;
    QJsonObject cStateToJson(const CConsciousnessState& state) const;
};

#endif // RUSTBRAINBRIDGE_H
