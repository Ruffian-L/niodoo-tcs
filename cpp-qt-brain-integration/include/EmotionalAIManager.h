#ifndef EMOTIONALAIMANAGER_H
#define EMOTIONALAIMANAGER_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QTimer>
#include <QMap>

class EmotionalAIManager : public QObject
{
    Q_OBJECT

public:
    explicit EmotionalAIManager(QObject *parent = nullptr);
    ~EmotionalAIManager();

    // Configuration
    void setArchitectEndpoint(const QString& endpoint);
    void setDeveloperEndpoint(const QString& endpoint);
    void setDistributedMode(bool enabled);
    void setTimeout(int milliseconds);
    void setMixedPrecision(bool enabled);
    bool useMixedPrecision() const;

    // Processing
    void processEmotionalInput(const QString& input);
    void analyzeComplexEmotions(const QString& scenario);
    void processTraumaInformedInput(const QString& input);

    // Results
    QString getLastAnalysis() const;
    QJsonObject getDetectedEmotions() const;
    QJsonObject getNeuralPathways() const;
    double getEmpathyScore() const;
    double getConsensusLevel() const;

    // Advanced features
    void calibrateEmotionalIntensity(const QString& emotion, double intensity);
    void updateCulturalContext(const QString& culture, const QJsonObject& parameters);
    void setTraumaSensitivity(bool enabled);

signals:
    void processingStarted();
    void processingCompleted();
    void emotionsDetected(const QJsonObject& emotions);
    void neuralPathwaysActivated(const QJsonObject& pathways);
    void empathyScoreUpdated(double score);
    void consensusLevelChanged(double level);
    void errorOccurred(const QString& error);

private slots:
    void onArchitectResponse(QNetworkReply* reply);
    void onDeveloperResponse(QNetworkReply* reply);
    void onBrainSystemResponse(QNetworkReply* reply);
    void onNetworkError(QNetworkReply::NetworkError error);
    void onProcessingTimeout();

private:
    // Network management
    QNetworkAccessManager* architectManager;
    QNetworkAccessManager* developerManager;
    QNetworkAccessManager* brainManager;

    // Configuration
    QString architectEndpoint;
    QString developerEndpoint;
    QString brainEndpoint;
    bool distributedMode;
    int timeoutMs;

    // Processing state
    QString currentInput;
    QString architectPlan;
    QString developerImplementation;
    bool processingInProgress;
    QTimer* processingTimer;

    // Results
    QString lastAnalysis;
    QJsonObject detectedEmotions;
    QJsonObject neuralPathways;
    double empathyScore;
    double consensusLevel;

    // Advanced settings
    QMap<QString, double> emotionalIntensityCalibration;
    QMap<QString, QJsonObject> culturalContexts;
    bool traumaSensitivityEnabled;
    bool m_useMixedPrecision = false;

    // Private methods
    void startProcessing();
    void completeProcessing();
    void sendToArchitect();
    void sendToDeveloper();
    void sendToBrainSystem();
    void parseEmotionalResponse(const QString& response);
    void parseBrainSystemResponse(const QString& response);
    void calculateEmpathyScore();
    void updateConsensusLevel();

    // Predefined emotion patterns
    QJsonObject getSuppressedEmotionPatterns();
    QJsonObject getComplexEmotionMatrix();
};

#endif // EMOTIONALAIMANAGER_H
