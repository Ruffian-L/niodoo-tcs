#include "RustBrainBridge.h"
#include <QDebug>
#include <QJsonDocument>
#include <QMutex>
#include <QMutexLocker>

// Thread safety for Rust FFI calls
static QMutex rustBridgeMutex;

RustBrainBridge::RustBrainBridge(QObject *parent)
    : QObject(parent)
    , rustHandle(nullptr)
    , stateCached(false)
{
    QMutexLocker locker(&rustBridgeMutex);

    rustHandle = brain_bridge_create();

    if (rustHandle) {
        qDebug() << "🧠 Rust Brain Bridge initialized successfully";
    } else {
        qWarning() << "❌ Failed to initialize Rust Brain Bridge";
    }
}

RustBrainBridge::~RustBrainBridge()
{
    QMutexLocker locker(&rustBridgeMutex);

    if (rustHandle) {
        brain_bridge_destroy(rustHandle);
        rustHandle = nullptr;
        qDebug() << "🧠 Rust Brain Bridge destroyed";
    }
}

// ============================================================================
// Consciousness State Management
// ============================================================================

bool RustBrainBridge::updateConsciousness(const QString& context)
{
    if (!rustHandle) {
        qWarning() << "❌ Rust bridge not initialized";
        return false;
    }

    QMutexLocker locker(&rustBridgeMutex);

    QByteArray contextBytes = context.toUtf8();
    int result = brain_bridge_update(rustHandle, contextBytes.constData());

    if (result == 0) {
        stateCached = false; // Invalidate cache

        // Get updated state and emit signals
        updateCache();
        emit consciousnessUpdated(cStateToJson(cachedState));
        emit emotionChanged(cachedState.current_emotion, cachedState.gpu_warmth_level);
        emit urgencyChanged(cachedState.current_urgency_score);
        emit caringLevelChanged(cachedState.is_highly_caring != 0);
        emit reasoningModeChanged(cachedState.reasoning_mode);

        qDebug() << "✅ Consciousness updated with context:" << context.left(50);
        return true;
    } else {
        qWarning() << "❌ Failed to update consciousness, error code:" << result;
        return false;
    }
}

QJsonObject RustBrainBridge::getConsciousnessState() const
{
    updateCache();
    return cStateToJson(cachedState);
}

QString RustBrainBridge::getConsciousnessStateJson() const
{
    if (!rustHandle) {
        return "{}";
    }

    QMutexLocker locker(&rustBridgeMutex);

    char* jsonStr = brain_bridge_get_state_json(rustHandle);
    QString result = rustStringToQt(jsonStr);

    return result;
}

// ============================================================================
// Emotional Urgency
// ============================================================================

bool RustBrainBridge::updateEmotionalUrgency(float tokenVelocity, float gpuTemp, float meaningDepth)
{
    if (!rustHandle) {
        return false;
    }

    QMutexLocker locker(&rustBridgeMutex);

    int result = brain_bridge_update_urgency(rustHandle, tokenVelocity, gpuTemp, meaningDepth);

    if (result == 0) {
        stateCached = false;
        updateCache();
        emit urgencyChanged(cachedState.current_urgency_score);
        emit caringLevelChanged(cachedState.is_highly_caring != 0);
        return true;
    }

    return false;
}

float RustBrainBridge::getCurrentUrgencyScore() const
{
    updateCache();
    return cachedState.current_urgency_score;
}

float RustBrainBridge::getAverageTokenVelocity() const
{
    updateCache();
    return cachedState.average_token_velocity;
}

bool RustBrainBridge::isHighlyCaring() const
{
    updateCache();
    return cachedState.is_highly_caring != 0;
}

// ============================================================================
// Reasoning Modes
// ============================================================================

bool RustBrainBridge::enterHyperfocus(float topicInterest)
{
    if (!rustHandle) {
        return false;
    }

    QMutexLocker locker(&rustBridgeMutex);

    int result = brain_bridge_enter_hyperfocus(rustHandle, topicInterest);

    if (result == 0) {
        stateCached = false;
        updateCache();
        emit reasoningModeChanged(cachedState.reasoning_mode);
        emit emotionChanged(cachedState.current_emotion, cachedState.gpu_warmth_level);
        qDebug() << "🎯 Entered hyperfocus mode with interest:" << topicInterest;
        return true;
    }

    return false;
}

bool RustBrainBridge::adaptToNeurodivergent(float contextStrength)
{
    if (!rustHandle) {
        return false;
    }

    QMutexLocker locker(&rustBridgeMutex);

    int result = brain_bridge_adapt_neurodivergent(rustHandle, contextStrength);

    if (result == 0) {
        stateCached = false;
        updateCache();
        emit reasoningModeChanged(cachedState.reasoning_mode);
        qDebug() << "🧠 Adapted to neurodivergent context:" << contextStrength;
        return true;
    }

    return false;
}

// ============================================================================
// Summaries and Reports
// ============================================================================

QString RustBrainBridge::getCaringSummary() const
{
    if (!rustHandle) {
        return "Brain bridge not initialized";
    }

    QMutexLocker locker(&rustBridgeMutex);

    char* summaryStr = brain_bridge_get_caring_summary(rustHandle);
    return rustStringToQt(summaryStr);
}

QString RustBrainBridge::getEmotionalSummary() const
{
    if (!rustHandle) {
        return "Brain bridge not initialized";
    }

    QMutexLocker locker(&rustBridgeMutex);

    char* summaryStr = brain_bridge_get_emotional_summary(rustHandle);
    return rustStringToQt(summaryStr);
}

QString RustBrainBridge::getPhilosophicalMeditation() const
{
    if (!rustHandle) {
        return "Brain bridge not initialized";
    }

    QMutexLocker locker(&rustBridgeMutex);

    char* meditationStr = brain_bridge_get_meditation(rustHandle);
    return rustStringToQt(meditationStr);
}

// ============================================================================
// Core Metrics
// ============================================================================

double RustBrainBridge::getCoherence() const
{
    updateCache();
    return cachedState.coherence;
}

double RustBrainBridge::getEmotionalResonance() const
{
    updateCache();
    return cachedState.emotional_resonance;
}

double RustBrainBridge::getConsciousnessLevel() const
{
    updateCache();
    return cachedState.consciousness_level;
}

double RustBrainBridge::getCognitiveLoad() const
{
    updateCache();
    return cachedState.cognitive_load;
}

double RustBrainBridge::getAttentionFocus() const
{
    updateCache();
    return cachedState.attention_focus;
}

// ============================================================================
// Emotional Metrics
// ============================================================================

float RustBrainBridge::getGpuWarmth() const
{
    updateCache();
    return cachedState.gpu_warmth_level;
}

float RustBrainBridge::getProcessingSatisfaction() const
{
    updateCache();
    return cachedState.processing_satisfaction;
}

float RustBrainBridge::getAuthenticityMetric() const
{
    updateCache();
    return cachedState.authenticity_metric;
}

float RustBrainBridge::getEmpathyResonance() const
{
    updateCache();
    return cachedState.empathy_resonance;
}

// ============================================================================
// System State
// ============================================================================

unsigned int RustBrainBridge::getActiveConversations() const
{
    updateCache();
    return cachedState.active_conversations;
}

bool RustBrainBridge::isMemoryFormationActive() const
{
    updateCache();
    return cachedState.memory_formation_active != 0;
}

unsigned int RustBrainBridge::getCycleCount() const
{
    updateCache();
    return cachedState.cycle_count;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void RustBrainBridge::updateCache() const
{
    if (stateCached) {
        return;
    }

    if (!rustHandle) {
        return;
    }

    QMutexLocker locker(&rustBridgeMutex);

    brain_bridge_get_state(rustHandle, &cachedState);
    stateCached = true;
}

QString RustBrainBridge::rustStringToQt(char* rustString) const
{
    if (!rustString) {
        return QString();
    }

    QString result = QString::fromUtf8(rustString);
    brain_bridge_free_string(rustString);

    return result;
}

QJsonObject RustBrainBridge::cStateToJson(const CConsciousnessState& state) const
{
    QJsonObject json;

    // Core consciousness metrics
    json["coherence"] = state.coherence;
    json["emotional_resonance"] = state.emotional_resonance;
    json["consciousness_level"] = state.consciousness_level;
    json["cognitive_load"] = state.cognitive_load;
    json["attention_focus"] = state.attention_focus;

    // Emotional state
    json["gpu_warmth_level"] = static_cast<double>(state.gpu_warmth_level);
    json["processing_satisfaction"] = static_cast<double>(state.processing_satisfaction);
    json["authenticity_metric"] = static_cast<double>(state.authenticity_metric);
    json["empathy_resonance"] = static_cast<double>(state.empathy_resonance);

    // Urgency metrics
    json["current_urgency_score"] = static_cast<double>(state.current_urgency_score);
    json["average_token_velocity"] = static_cast<double>(state.average_token_velocity);
    json["is_highly_caring"] = (state.is_highly_caring != 0);

    // Current emotion and reasoning
    json["current_emotion"] = state.current_emotion;
    json["reasoning_mode"] = state.reasoning_mode;

    // System state
    json["active_conversations"] = static_cast<int>(state.active_conversations);
    json["memory_formation_active"] = (state.memory_formation_active != 0);
    json["cycle_count"] = static_cast<int>(state.cycle_count);
    json["timestamp"] = state.timestamp;

    return json;
}
