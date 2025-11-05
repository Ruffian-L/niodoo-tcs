#include "BrainSystemBridge.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonArray>
#include <QThread>
#include <QDateTime>
#include <QRandomGenerator>

BrainSystemBridge::BrainSystemBridge(QObject *parent)
    : QObject(parent)
    , networkManager(new QNetworkAccessManager(this))
    , brainEndpoint("ws://localhost:8080")
    , updateTimer(new QTimer(this))
    , agentsCount(11)
    , connectionsCount(25)
    , pathwayActivation(0.7)
    , consensusThreshold(0.6)
    , updateIntervalMs(1000)
    , systemRunning(false)
    , currentConsensusLevel(0.0)
    , currentActivityLevel(0.0)
{
    // Connect network signals
    connect(networkManager, &QNetworkAccessManager::finished, this, [this](QNetworkReply* reply) {
        reply->deleteLater();
    });

    // Setup update timer
    connect(updateTimer, &QTimer::timeout, this, &BrainSystemBridge::onStatusUpdate);

    // Initialize default system
    initializeDefaultAgents();
    createDefaultPathways();

    qDebug() << "🧠 BrainSystemBridge initialized with" << agentsCount << "agents";
}

BrainSystemBridge::~BrainSystemBridge()
{
    stopBrainSystem();
    qDebug() << "🧠 BrainSystemBridge destroyed";
}

// Configuration
void BrainSystemBridge::setAgentsCount(int count)
{
    agentsCount = qMax(1, count);
    qDebug() << "🔧 Agents count set to:" << agentsCount;
}

void BrainSystemBridge::setConnectionsCount(int count)
{
    connectionsCount = qMax(1, count);
    qDebug() << "🔧 Connections count set to:" << connectionsCount;
}

void BrainSystemBridge::setBrainEndpoint(const QString& endpoint)
{
    brainEndpoint = endpoint;
    qDebug() << "🔧 Brain endpoint set to:" << brainEndpoint;
}

void BrainSystemBridge::setPathwayActivation(double level)
{
    pathwayActivation = qBound(0.0, level, 1.0);
    qDebug() << "🔧 Pathway activation set to:" << pathwayActivation;
}

void BrainSystemBridge::setUpdateInterval(int milliseconds)
{
    updateIntervalMs = qMax(100, milliseconds);
    updateTimer->setInterval(updateIntervalMs);
    qDebug() << "🔧 Update interval set to:" << updateIntervalMs << "ms";
}

// System control
void BrainSystemBridge::startBrainSystem()
{
    if (systemRunning) {
        qDebug() << "⚠️ Brain system already running";
        return;
    }

    systemRunning = true;
    updateTimer->start(updateIntervalMs);
    
    qDebug() << "🚀 Brain system started";
    qDebug() << "   Endpoint:" << brainEndpoint;
    qDebug() << "   Agents:" << agentsCount;
    qDebug() << "   Update interval:" << updateIntervalMs << "ms";
}

void BrainSystemBridge::stopBrainSystem()
{
    if (!systemRunning) {
        return;
    }

    systemRunning = false;
    updateTimer->stop();
    
    qDebug() << "🛑 Brain system stopped";
}

void BrainSystemBridge::optimizePathways()
{
    qDebug() << "⚡ Optimizing neural pathways...";
    
    // Simulate optimization
    QThread::msleep(200);
    
    // Update pathway activation
    for (NeuralPathway& pathway : neuralPathways) {
        pathway.activationLevel = qMin(1.0, pathway.activationLevel + 0.1);
        pathway.isOptimized = true;
    }
    
    emit pathwaysUpdated(neuralPathways);
    emit systemOptimized();
    
    qDebug() << "✅ Pathway optimization complete";
}

void BrainSystemBridge::recalibrateAgents()
{
    qDebug() << "🔄 Recalibrating neural agents...";
    
    // Simulate recalibration
    QThread::msleep(150);
    
    // Update agent states
    for (NeuralAgent& agent : neuralAgents) {
        agent.activityLevel = qBound(0.0, agent.activityLevel + (QRandomGenerator::global()->generateDouble() - 0.5) * 0.2, 1.0);
        agent.lastActivation = QDateTime::currentDateTime().toString(Qt::ISODate);
    }
    
    emit agentsUpdated(neuralAgents);
    
    qDebug() << "✅ Agent recalibration complete";
}

// Data access
int BrainSystemBridge::getAgentsCount() const
{
    return agentsCount;
}

int BrainSystemBridge::getConnectionsCount() const
{
    return connectionsCount;
}

double BrainSystemBridge::getConsensusLevel() const
{
    return currentConsensusLevel;
}

double BrainSystemBridge::getActivityLevel() const
{
    return currentActivityLevel;
}

double BrainSystemBridge::getPathwayActivation() const
{
    return pathwayActivation;
}

QJsonObject BrainSystemBridge::getSystemStatus() const
{
    QJsonObject status;
    status["system_running"] = systemRunning;
    status["agents_count"] = agentsCount;
    status["connections_count"] = connectionsCount;
    status["consensus_level"] = currentConsensusLevel;
    status["activity_level"] = currentActivityLevel;
    status["pathway_activation"] = pathwayActivation;
    status["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    return status;
}

QVector<NeuralAgent> BrainSystemBridge::getActiveAgents() const
{
    QVector<NeuralAgent> activeAgents;
    for (const NeuralAgent& agent : neuralAgents) {
        if (agent.isActive) {
            activeAgents.append(agent);
        }
    }
    return activeAgents;
}

QVector<NeuralPathway> BrainSystemBridge::getActivePathways() const
{
    QVector<NeuralPathway> activePathways;
    for (const NeuralPathway& pathway : neuralPathways) {
        if (pathway.activationLevel > 0.1) {
            activePathways.append(pathway);
        }
    }
    return activePathways;
}

// Advanced features
void BrainSystemBridge::addNeuralAgent(const QString& type, const QJsonObject& config)
{
    NeuralAgent agent = createAgent(type, QString("agent_%1").arg(neuralAgents.size()));
    agent.connections = config;
    neuralAgents.append(agent);
    
    qDebug() << "➕ Added neural agent:" << agent.id << "(" << type << ")";
    emit agentsUpdated(neuralAgents);
}

void BrainSystemBridge::addPathway(const QString& name, const QStringList& agentIds)
{
    NeuralPathway pathway;
    pathway.id = QString("pathway_%1").arg(neuralPathways.size());
    pathway.name = name;
    pathway.activationLevel = pathwayActivation;
    pathway.agentCount = agentIds.size();
    pathway.isOptimized = false;
    
    // Create metadata
    QJsonObject metadata;
    metadata["agent_ids"] = QJsonArray::fromStringList(agentIds);
    metadata["created"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    pathway.metadata = metadata;
    
    neuralPathways.append(pathway);
    
    qDebug() << "➕ Added neural pathway:" << pathway.name << "with" << agentIds.size() << "agents";
    emit pathwaysUpdated(neuralPathways);
}

void BrainSystemBridge::updateAgentActivity(const QString& agentId, double activity)
{
    for (NeuralAgent& agent : neuralAgents) {
        if (agent.id == agentId) {
            agent.activityLevel = qBound(0.0, activity, 1.0);
            agent.lastActivation = QDateTime::currentDateTime().toString(Qt::ISODate);
            break;
        }
    }
    
    emit agentsUpdated(neuralAgents);
}

void BrainSystemBridge::setConsensusThreshold(double threshold)
{
    consensusThreshold = qBound(0.0, threshold, 1.0);
    qDebug() << "🔧 Consensus threshold set to:" << consensusThreshold;
}

// Private slots
void BrainSystemBridge::onHealthCheckResponse(QNetworkReply* reply)
{
    if (reply->error() == QNetworkReply::NoError) {
        QString response = reply->readAll();
        parseSystemStatus(response);
    } else {
        qDebug() << "❌ Health check failed:" << reply->errorString();
    }
}

void BrainSystemBridge::onAnalysisResponse(QNetworkReply* reply)
{
    if (reply->error() == QNetworkReply::NoError) {
        QString response = reply->readAll();
        qDebug() << "📊 Analysis response received:" << response.left(100);
    } else {
        qDebug() << "❌ Analysis request failed:" << reply->errorString();
    }
}

void BrainSystemBridge::onEmbeddingResponse(QNetworkReply* reply)
{
    if (reply->error() == QNetworkReply::NoError) {
        QString response = reply->readAll();
        qDebug() << "🔗 Embedding response received:" << response.left(100);
    } else {
        qDebug() << "❌ Embedding request failed:" << reply->errorString();
    }
}

void BrainSystemBridge::onVectorSearchResponse(QNetworkReply* reply)
{
    if (reply->error() == QNetworkReply::NoError) {
        QString response = reply->readAll();
        qDebug() << "🔍 Vector search response received:" << response.left(100);
    } else {
        qDebug() << "❌ Vector search request failed:" << reply->errorString();
    }
}

void BrainSystemBridge::onStatusUpdate()
{
    if (!systemRunning) {
        return;
    }

    updateSystemMetrics();
    calculateConsensus();
    updateActivityLevels();
    
    // Emit signals
    emit consensusChanged(currentConsensusLevel);
    emit activityChanged(currentActivityLevel);
}

void BrainSystemBridge::onNetworkError(QNetworkReply::NetworkError error)
{
    qDebug() << "❌ Network error:" << error;
    emit errorOccurred(QString("Network error: %1").arg(error));
}

// Private methods
void BrainSystemBridge::initializeDefaultAgents()
{
    neuralAgents.clear();
    
    QStringList agentTypes = {
        "Motor", "LCARS", "Efficiency", "Intuitive", "Analyst", 
        "Creative", "Logical", "Empathetic", "Strategic", "Tactical", "Adaptive"
    };
    
    for (int i = 0; i < qMin(agentsCount, agentTypes.size()); ++i) {
        NeuralAgent agent = createAgent(agentTypes[i], QString("agent_%1").arg(i));
        neuralAgents.append(agent);
    }
    
    qDebug() << "🧠 Initialized" << neuralAgents.size() << "default agents";
}

void BrainSystemBridge::createDefaultPathways()
{
    neuralPathways.clear();
    
    // Create empathy pathway
    QStringList empathyAgents = {"Empathetic", "Intuitive", "Creative"};
    addPathway("Empathy Circuit", empathyAgents);
    
    // Create reasoning pathway
    QStringList reasoningAgents = {"Analyst", "Logical", "Strategic"};
    addPathway("Reasoning Circuit", reasoningAgents);
    
    // Create action pathway
    QStringList actionAgents = {"Motor", "Tactical", "Efficiency"};
    addPathway("Action Circuit", actionAgents);
    
    qDebug() << "🧠 Created" << neuralPathways.size() << "default pathways";
}

void BrainSystemBridge::performHealthCheck()
{
    // Simulate health check
    QJsonObject healthStatus;
    healthStatus["status"] = "healthy";
    healthStatus["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    QJsonDocument doc(healthStatus);
    parseSystemStatus(doc.toJson());
}

void BrainSystemBridge::updateSystemMetrics()
{
    // Update agent states
    for (NeuralAgent& agent : neuralAgents) {
        agent.activityLevel = qBound(0.0, agent.activityLevel + (QRandomGenerator::global()->generateDouble() - 0.5) * 0.1, 1.0);
    }
    
    // Update pathway states
    for (NeuralPathway& pathway : neuralPathways) {
        pathway.activationLevel = qBound(0.0, pathway.activationLevel + (QRandomGenerator::global()->generateDouble() - 0.5) * 0.05, 1.0);
    }
}

void BrainSystemBridge::calculateConsensus()
{
    if (neuralAgents.isEmpty()) {
        currentConsensusLevel = 0.0;
        return;
    }
    
    double totalActivity = 0.0;
    for (const NeuralAgent& agent : neuralAgents) {
        totalActivity += agent.activityLevel;
    }
    
    currentConsensusLevel = totalActivity / neuralAgents.size();
}

void BrainSystemBridge::updateActivityLevels()
{
    if (neuralPathways.isEmpty()) {
        currentActivityLevel = 0.0;
        return;
    }
    
    double totalActivation = 0.0;
    for (const NeuralPathway& pathway : neuralPathways) {
        totalActivation += pathway.activationLevel;
    }
    
    currentActivityLevel = totalActivation / neuralPathways.size();
}

void BrainSystemBridge::sendAnalysisRequest(const QString& context)
{
    // Simulate analysis request
    QJsonObject request;
    request["type"] = "analysis";
    request["context"] = context;
    request["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    qDebug() << "📊 Sending analysis request for:" << context;
}

void BrainSystemBridge::sendEmbeddingRequest(const QString& text)
{
    // Simulate embedding request
    QJsonObject request;
    request["type"] = "embedding";
    request["text"] = text;
    request["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    qDebug() << "🔗 Sending embedding request for:" << text.left(50);
}

void BrainSystemBridge::sendVectorSearchRequest(const QString& query)
{
    // Simulate vector search request
    QJsonObject request;
    request["type"] = "vector_search";
    request["query"] = query;
    request["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    qDebug() << "🔍 Sending vector search request for:" << query;
}

void BrainSystemBridge::parseSystemStatus(const QString& response)
{
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8(), &error);
    
    if (error.error == QJsonParseError::NoError) {
        QJsonObject status = doc.object();
        qDebug() << "📊 System status parsed:" << status["status"].toString();
    } else {
        qDebug() << "❌ Failed to parse system status:" << error.errorString();
    }
}

void BrainSystemBridge::updateAgentStates()
{
    // Update agent states based on current system state
    for (NeuralAgent& agent : neuralAgents) {
        agent.isActive = agent.activityLevel > 0.1;
    }
}

void BrainSystemBridge::optimizeNeuralPathways()
{
    // Optimize pathways based on current consensus
    for (NeuralPathway& pathway : neuralPathways) {
        if (currentConsensusLevel > consensusThreshold) {
            pathway.activationLevel = qMin(1.0, pathway.activationLevel + 0.05);
        }
    }
}

// Agent management
NeuralAgent BrainSystemBridge::createAgent(const QString& type, const QString& id)
{
    NeuralAgent agent;
    agent.id = id;
    agent.type = type;
    agent.activityLevel = QRandomGenerator::global()->generateDouble();
    agent.isActive = agent.activityLevel > 0.3;
    agent.lastActivation = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    // Create connections metadata
    QJsonObject connections;
    connections["type"] = type;
    connections["created"] = agent.lastActivation;
    agent.connections = connections;
    
    return agent;
}

void BrainSystemBridge::activateAgent(const QString& agentId, const QString& reason)
{
    for (NeuralAgent& agent : neuralAgents) {
        if (agent.id == agentId) {
            agent.isActive = true;
            agent.activityLevel = qMin(1.0, agent.activityLevel + 0.2);
            agent.lastActivation = QDateTime::currentDateTime().toString(Qt::ISODate);
            qDebug() << "✅ Activated agent:" << agentId << "(" << reason << ")";
            break;
        }
    }
}

void BrainSystemBridge::deactivateAgent(const QString& agentId)
{
    for (NeuralAgent& agent : neuralAgents) {
        if (agent.id == agentId) {
            agent.isActive = false;
            agent.activityLevel = qMax(0.0, agent.activityLevel - 0.2);
            qDebug() << "❌ Deactivated agent:" << agentId;
            break;
        }
    }
}

double BrainSystemBridge::calculateAgentActivity(const NeuralAgent& agent)
{
    return agent.activityLevel;
}

bool BrainSystemBridge::checkPathwayOptimization(const NeuralPathway& pathway)
{
    return pathway.activationLevel > 0.8 && pathway.isOptimized;
}

// Consensus algorithms
double BrainSystemBridge::calculateConsensus(const QVector<NeuralAgent>& agents)
{
    if (agents.isEmpty()) {
        return 0.0;
    }
    
    double totalActivity = 0.0;
    for (const NeuralAgent& agent : agents) {
        totalActivity += agent.activityLevel;
    }
    
    return totalActivity / agents.size();
}

double BrainSystemBridge::calculatePathwayConsensus(const NeuralPathway& pathway)
{
    return pathway.activationLevel;
}

void BrainSystemBridge::updateAgentConsensus(NeuralAgent& agent)
{
    // Update agent consensus based on pathway activation
    agent.activityLevel = qBound(0.0, agent.activityLevel + (pathwayActivation - 0.5) * 0.1, 1.0);
}