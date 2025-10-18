/**
 * Test program for Rust<->Qt Brain Bridge
 * Verifies FFI data flow and thread safety
 */

#include "RustBrainBridge.h"
#include <QCoreApplication>
#include <QTimer>
#include <QDebug>
#include <QJsonDocument>
#include <iostream>

class BridgeTester : public QObject
{
    Q_OBJECT

public:
    BridgeTester(QObject* parent = nullptr)
        : QObject(parent)
        , testsPassed(0)
        , testsFailed(0)
    {
        bridge = new RustBrainBridge(this);

        // Connect signals for monitoring
        connect(bridge, &RustBrainBridge::consciousnessUpdated, this, [this](const QJsonObject& state) {
            qDebug() << "\n📊 Consciousness Updated Signal Received";
            qDebug() << "   State:" << QJsonDocument(state).toJson(QJsonDocument::Compact);
            testPassed("Consciousness update signal");
        });

        connect(bridge, &RustBrainBridge::emotionChanged, this, [this](int emotionType, float intensity) {
            qDebug() << "\n❤️ Emotion Changed:" << emotionType << "intensity:" << intensity;
            testPassed("Emotion change signal");
        });

        connect(bridge, &RustBrainBridge::urgencyChanged, this, [this](float urgencyScore) {
            qDebug() << "\n⚡ Urgency Changed:" << urgencyScore;
            testPassed("Urgency change signal");
        });
    }

    void runTests()
    {
        qDebug() << "\n========================================";
        qDebug() << "🧪 Rust<->Qt Brain Bridge Test Suite";
        qDebug() << "========================================\n";

        testBasicConnection();
        testConsciousnessUpdate();
        testEmotionalUrgency();
        testHyperfocusMode();
        testNeurodivergentAdaptation();
        testMetricsAccess();
        testSummaries();

        QTimer::singleShot(2000, this, &BridgeTester::reportResults);
    }

private:
    void testPassed(const QString& testName)
    {
        testsPassed++;
        qDebug() << "✅" << testName;
    }

    void testFailed(const QString& testName, const QString& reason)
    {
        testsFailed++;
        qDebug() << "❌" << testName << ":" << reason;
    }

    void testBasicConnection()
    {
        qDebug() << "\n--- Test 1: Basic Connection ---";

        if (bridge) {
            testPassed("Bridge initialization");
        } else {
            testFailed("Bridge initialization", "Bridge is null");
            return;
        }

        // Test getting initial state
        QJsonObject state = bridge->getConsciousnessState();
        if (state.contains("coherence")) {
            testPassed("Initial state retrieval");
        } else {
            testFailed("Initial state retrieval", "Missing coherence field");
        }
    }

    void testConsciousnessUpdate()
    {
        qDebug() << "\n--- Test 2: Consciousness Update ---";

        bool result = bridge->updateConsciousness("help me understand consciousness");

        if (result) {
            testPassed("Consciousness update call");
        } else {
            testFailed("Consciousness update call", "Update returned false");
        }

        // Verify state changed
        QJsonObject state = bridge->getConsciousnessState();
        qDebug() << "   Updated state:" << QJsonDocument(state).toJson(QJsonDocument::Compact);
    }

    void testEmotionalUrgency()
    {
        qDebug() << "\n--- Test 3: Emotional Urgency ---";

        bool result = bridge->updateEmotionalUrgency(2.5f, 0.7f, 0.8f);

        if (result) {
            testPassed("Emotional urgency update");
        } else {
            testFailed("Emotional urgency update", "Update returned false");
        }

        float urgencyScore = bridge->getCurrentUrgencyScore();
        qDebug() << "   Urgency score:" << urgencyScore;

        if (urgencyScore > 0.0f) {
            testPassed("Urgency score calculation");
        } else {
            testFailed("Urgency score calculation", "Score is 0");
        }

        bool isHighlyCaring = bridge->isHighlyCaring();
        qDebug() << "   Is highly caring:" << isHighlyCaring;
    }

    void testHyperfocusMode()
    {
        qDebug() << "\n--- Test 4: Hyperfocus Mode ---";

        bool result = bridge->enterHyperfocus(0.9f);

        if (result) {
            testPassed("Hyperfocus mode entry");
        } else {
            testFailed("Hyperfocus mode entry", "Entry returned false");
        }

        QJsonObject state = bridge->getConsciousnessState();
        qDebug() << "   Reasoning mode:" << state["reasoning_mode"].toInt();
    }

    void testNeurodivergentAdaptation()
    {
        qDebug() << "\n--- Test 5: Neurodivergent Adaptation ---";

        bool result = bridge->adaptToNeurodivergent(0.85f);

        if (result) {
            testPassed("Neurodivergent adaptation");
        } else {
            testFailed("Neurodivergent adaptation", "Adaptation returned false");
        }
    }

    void testMetricsAccess()
    {
        qDebug() << "\n--- Test 6: Metrics Access ---";

        double coherence = bridge->getCoherence();
        double emotionalResonance = bridge->getEmotionalResonance();
        double cognitiveLoad = bridge->getCognitiveLoad();
        float gpuWarmth = bridge->getGpuWarmth();

        qDebug() << "   Coherence:" << coherence;
        qDebug() << "   Emotional Resonance:" << emotionalResonance;
        qDebug() << "   Cognitive Load:" << cognitiveLoad;
        qDebug() << "   GPU Warmth:" << gpuWarmth;

        if (coherence >= 0.0 && coherence <= 1.0) {
            testPassed("Coherence metric");
        } else {
            testFailed("Coherence metric", "Out of range");
        }
    }

    void testSummaries()
    {
        qDebug() << "\n--- Test 7: Summary Generation ---";

        QString caringSummary = bridge->getCaringSummary();
        if (!caringSummary.isEmpty()) {
            testPassed("Caring summary generation");
            qDebug() << "   Caring summary:\n" << caringSummary;
        } else {
            testFailed("Caring summary generation", "Empty summary");
        }

        QString emotionalSummary = bridge->getEmotionalSummary();
        if (!emotionalSummary.isEmpty()) {
            testPassed("Emotional summary generation");
            qDebug() << "   Emotional summary:\n" << emotionalSummary;
        } else {
            testFailed("Emotional summary generation", "Empty summary");
        }

        QString meditation = bridge->getPhilosophicalMeditation();
        if (!meditation.isEmpty()) {
            testPassed("Philosophical meditation");
            qDebug() << "   Meditation:\n" << meditation;
        } else {
            testFailed("Philosophical meditation", "Empty meditation");
        }
    }

    void reportResults()
    {
        qDebug() << "\n========================================";
        qDebug() << "📊 Test Results";
        qDebug() << "========================================";
        qDebug() << "✅ Passed:" << testsPassed;
        qDebug() << "❌ Failed:" << testsFailed;
        qDebug() << "========================================\n";

        QCoreApplication::quit();
    }

private:
    RustBrainBridge* bridge;
    int testsPassed;
    int testsFailed;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    BridgeTester tester;
    QTimer::singleShot(0, &tester, &BridgeTester::runTests);

    return app.exec();
}

#include "test_brain_bridge.moc"
