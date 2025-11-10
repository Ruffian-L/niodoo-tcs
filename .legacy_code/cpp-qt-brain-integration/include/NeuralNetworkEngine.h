#ifndef NEURALNETWORKENGINE_H
#define NEURALNETWORKENGINE_H

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QVector>
#include <QMap>
#include <QTimer>

struct HardwareMetrics {
    double gpuUsage;
    double memoryUsage;
    double temperature;
    int powerConsumption;
    QString deviceName;
};

class NeuralNetworkEngine : public QObject
{
    Q_OBJECT

public:
    explicit NeuralNetworkEngine(QObject *parent = nullptr);
    ~NeuralNetworkEngine();

    // Hardware monitoring
    double getGPUUsage() const;
    double getMemoryUsage() const;
    double getTemperature() const;
    int getPowerConsumption() const;
    QString getDeviceName() const;

    // Performance optimization
    void setHardwareAcceleration(bool enabled);
    void setMaxThreads(int threads);
    void optimizeForHardware();
    void updateMetrics();

    // Mock inference (no ONNX Runtime dependency)
    QJsonObject runEmotionalInference(const QString& input);
    QJsonObject runConsensusInference(const QVector<double>& agentActivities);
    QJsonObject runEmbeddingInference(const QString& text);
    QJsonObject runVectorSearchInference(const QString& query, int topK = 10);

    // ONNX methods
    void initializeONNXRuntime();
    void loadEmotionalModel(const QString& modelPath);
    void loadConsensusModel(const QString& modelPath);
    bool isONNXInitialized() const;
    void setOptimizationLevel(int level);
    void enableMixedPrecision(bool enabled);
    void setMemoryLimit(size_t limit);

signals:
    void hardwareMetricsUpdated(const HardwareMetrics& metrics);
    void inferenceCompleted(const QString& modelType, const QJsonObject& result);

private slots:
    void onMetricsUpdate();

private:
    // Hardware monitoring
    QTimer* metricsTimer;
    HardwareMetrics currentMetrics;
    bool hardwareAcceleration;
    int maxThreads;

    // Mock implementation
    void updateHardwareMetrics();
    double calculateGPUUsage();
    double calculateMemoryUsage();
    double calculateTemperature();
    int calculatePowerConsumption();

    QJsonObject mockEmotionalInference(const QString& input);
    QJsonObject mockConsensusInference(const QVector<double>& activities);
    QJsonObject mockEmbeddingInference(const QString& text);
    QJsonObject mockVectorSearchInference(const QString& query, int topK);

#ifdef HAS_ONNXRUNTIME
    Ort::Env m_env;
    Ort::Session m_emotionalSession{nullptr};
    Ort::Session m_consensusSession{nullptr};
    QString m_emotionalModelPath;
    QString m_consensusModelPath;
    bool m_onnxInitialized = false;
#endif
};

#endif // NEURALNETWORKENGINE_H