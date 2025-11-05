#ifndef BRAINSYSTEMBRIDGE_H
#define BRAINSYSTEMBRIDGE_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QTimer>
#include <QVector>
#include <QMap>

struct NeuralAgent {
    QString id;
    QString type;
    double activityLevel;
    QJsonObject connections;
    bool isActive;
    QString lastActivation;
};

struct NeuralPathway {
    QString id;
    QString name;
    double activationLevel;
    int agentCount;
    QJsonObject metadata;
    bool isOptimized;
};

class BrainSystemBridge : public QObject
{
    Q_OBJECT

public:
    explicit BrainSystemBridge(QObject *parent = nullptr);
    ~BrainSystemBridge();

    // Configuration
    void setAgentsCount(int count);
    void setConnectionsCount(int count);
    void setBrainEndpoint(const QString& endpoint);
    void setPathwayActivation(double level);
    void setUpdateInterval(int milliseconds);

    // System control
    void startBrainSystem();
    void stopBrainSystem();
    void optimizePathways();
    void recalibrateAgents();

    // Data access
    int getAgentsCount() const;
    int getConnectionsCount() const;
    double getConsensusLevel() const;
    double getActivityLevel() const;
    double getPathwayActivation() const;
    QJsonObject getSystemStatus() const;
    QVector<NeuralAgent> getActiveAgents() const;
    QVector<NeuralPathway> getActivePathways() const;

    // Advanced features
    void addNeuralAgent(const QString& type, const QJsonObject& config);
    void addPathway(const QString& name, const QStringList& agentIds);
    void updateAgentActivity(const QString& agentId, double activity);
    void setConsensusThreshold(double threshold);

signals:
    void agentsUpdated(const QVector<NeuralAgent>& agents);
    void pathwaysUpdated(const QVector<NeuralPathway>& pathways);
    void consensusChanged(double level);
    void activityChanged(double level);
    void systemOptimized();
    void errorOccurred(const QString& error);

private slots:
    void onHealthCheckResponse(QNetworkReply* reply);
    void onAnalysisResponse(QNetworkReply* reply);
    void onEmbeddingResponse(QNetworkReply* reply);
    void onVectorSearchResponse(QNetworkReply* reply);
    void onStatusUpdate();
    void onNetworkError(QNetworkReply::NetworkError error);

private:
    // Network management
    QNetworkAccessManager* networkManager;
    QString brainEndpoint;
    QTimer* updateTimer;

    // System state
    int agentsCount;
    int connectionsCount;
    double pathwayActivation;
    double consensusThreshold;
    int updateIntervalMs;
    bool systemRunning;

    // Neural structures
    QVector<NeuralAgent> neuralAgents;
    QVector<NeuralPathway> neuralPathways;
    QMap<QString, double> agentActivityMap;
    QMap<QString, QJsonObject> agentConnections;

    // System metrics
    double currentConsensusLevel;
    double currentActivityLevel;
    QJsonObject lastSystemStatus;

    // Private methods
    void initializeDefaultAgents();
    void createDefaultPathways();
    void performHealthCheck();
    void updateSystemMetrics();
    void calculateConsensus();
    void updateActivityLevels();
    void sendAnalysisRequest(const QString& context);
    void sendEmbeddingRequest(const QString& text);
    void sendVectorSearchRequest(const QString& query);
    void parseSystemStatus(const QString& response);
    void updateAgentStates();
    void optimizeNeuralPathways();

    // Agent management
    NeuralAgent createAgent(const QString& type, const QString& id);
    // NeuralPathway createPathway(const QString& name, const QStringList& agentIds); // Removed due to conflict
    void activateAgent(const QString& agentId, const QString& reason);
    void deactivateAgent(const QString& agentId);
    double calculateAgentActivity(const NeuralAgent& agent);
    bool checkPathwayOptimization(const NeuralPathway& pathway);

    // Consensus algorithms
    double calculateConsensus(const QVector<NeuralAgent>& agents);
    double calculatePathwayConsensus(const NeuralPathway& pathway);
    void updateAgentConsensus(NeuralAgent& agent);
};

#endif // BRAINSYSTEMBRIDGE_H
