#include "MainWindow.h"
#include <QApplication>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QTextEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QListWidget>
#include <QGroupBox>
#include <QSpinBox>
#include <QSlider>
#include <QCheckBox>
#include <QComboBox>
#include <QSplitter>
#include <QScrollArea>
#include <QTimer>
#include <QSystemTrayIcon>
#include <QIcon>
#include <QPixmap>
#include <QProcessEnvironment>
#include <QCloseEvent>
#include <QFile>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , emotionalAI(nullptr)
    , brainBridge(nullptr)
    , neuralEngine(nullptr)
    , statusTimer(new QTimer(this))
    , trayIcon(nullptr)
    , settings(new QSettings("Niodoo", "BrainIntegration", this))
    , distributedMode(true)
    , agentsCount(0)
    , connectionsCount(0)
{
    // Initialize endpoints from environment variables
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    architectEndpoint = env.value("NIODOO_ARCHITECT_ENDPOINT", "http://localhost:11434/api/generate");
    developerEndpoint = env.value("NIODOO_DEVELOPER_ENDPOINT", "http://localhost:11434/api/generate");
    setWindowTitle("Brain Integration - Emotional AI System");
    setMinimumSize(1200, 800);
    resize(1400, 900);

    // Load configuration from settings
    loadConfiguration();

    // Initialize core systems
    initializeComponents();

    // Setup UI
    setupUI();
    setupMenus();
    setupStatusBar();
    connectSignals();

    // Start monitoring
    statusTimer->start(1000); // Update every second

    // Setup system tray
    setupTrayIcon();
}

void MainWindow::loadConfiguration()
{
    // Load core settings
    emotionThreshold = settings->value("core/emotion_threshold", 0.7).toDouble();
    maxHistory = settings->value("core/max_history", 50).toInt();
    dbPath = settings->value("core/db_path", "data/knowledge_graph.db").toString();
    backupInterval = settings->value("core/backup_interval", 3600).toInt();
    contextWindow = settings->value("core/context_window", 10).toInt();
    responseDelay = settings->value("core/response_delay", 0.5).toDouble();

    // Load model settings
    modelName = settings->value("models/default_model", "llama3:latest").toString();
    backupModelName = settings->value("models/backup_model", "llama3.2:3b").toString();
    modelTemperature = settings->value("models/temperature", 0.8).toDouble();
    maxTokens = settings->value("models/max_tokens", 200).toInt();
    modelTimeout = settings->value("models/timeout", 30).toInt();
    topP = settings->value("models/top_p", 0.9).toDouble();
    topK = settings->value("models/top_k", 40).toInt();
    repeatPenalty = settings->value("models/repeat_penalty", 1.1).toDouble();
    frequencyPenalty = settings->value("models/frequency_penalty", 0.1).toDouble();
    presencePenalty = settings->value("models/presence_penalty", 0.1).toDouble();

    // Load Qt/GUI settings
    qtEmotionThreshold = settings->value("qt/emotion_threshold", 0.7).toDouble();
    distributedMode = settings->value("qt/distributed_mode", true).toBool();
    // Use environment variables as primary source, config file as fallback, no hardcoded defaults
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    // Try to load from config.toml first - fail loudly if missing required values
    QString configPath = QDir::currentPath() + "/config.toml";
    bool configLoaded = false;

    if (QFile::exists(configPath)) {
        QSettings configSettings(configPath, QSettings::IniFormat);

        if (configSettings.contains("qt/agents_count") && configSettings.contains("qt/connections_count")) {
            int configAgentsCount = configSettings.value("qt/agents_count").toInt();
            int configConnectionsCount = configSettings.value("qt/connections_count").toInt();
            configLoaded = true;

            qDebug() << "Loaded Qt config from" << configPath;
            qDebug() << "Agents count:" << configAgentsCount;
            qDebug() << "Connections count:" << configConnectionsCount;
        } else {
            qCritical() << "ERROR: Config file missing required [qt] section values: agents_count, connections_count";
            qCritical() << "Config file path:" << configPath;
            qCritical() << "Please check your TOML config, fool!";
        }
    } else {
        qWarning() << "Config file not found at" << configPath;
    }

    // Try environment variables if config file didn't have the values
    if (!configLoaded) {
        QString envAgents = env.value("NIODOO_AGENTS_COUNT");
        QString envConnections = env.value("NIODOO_CONNECTIONS_COUNT");

        if (!envAgents.isEmpty() && !envConnections.isEmpty()) {
            agentsCount = envAgents.toInt();
            connectionsCount = envConnections.toInt();
            configLoaded = true;

            qDebug() << "Loaded Qt config from environment variables";
            qDebug() << "Agents count:" << agentsCount;
            qDebug() << "Connections count:" << connectionsCount;
        } else {
            qCritical() << "ERROR: No valid Qt configuration found!";
            qCritical() << "Either provide config.toml with [qt] section or set NIODOO_AGENTS_COUNT and NIODOO_CONNECTIONS_COUNT environment variables";
            qCritical() << "Qt integration cannot initialize without proper config";
            // Keep existing values (which should be 0 from initialization) to indicate configuration failure
        }
    }

    architectEndpoint = env.value("NIODOO_ARCHITECT_ENDPOINT", settings->value("qt/architect_endpoint", "http://localhost:11434/api/generate").toString());
    developerEndpoint = env.value("NIODOO_DEVELOPER_ENDPOINT", settings->value("qt/developer_endpoint", "http://localhost:11434/api/generate").toString());
    hardwareAcceleration = settings->value("qt/hardware_acceleration", true).toBool();
    networkMode = settings->value("qt/network_mode", "Distributed").toString();
    pathwayActivation = settings->value("qt/pathway_activation", 100).toInt();

    // Load API settings
    ollamaUrl = settings->value("api/ollama_url", "http://localhost:11434").toString();
    apiTimeout = settings->value("api/api_timeout", 30).toInt();
    retryAttempts = settings->value("api/retry_attempts", 3).toInt();
    enableCaching = settings->value("api/enable_caching", true).toBool();
    cacheTtl = settings->value("api/cache_ttl", 300).toInt();

    // Load consciousness settings
    consciousnessEnabled = settings->value("consciousness/enabled", true).toBool();
    reflectionEnabled = settings->value("consciousness/reflection_enabled", true).toBool();
    emotionSensitivity = settings->value("consciousness/emotion_sensitivity", 0.8).toDouble();
    memoryThreshold = settings->value("consciousness/memory_threshold", 0.6).toDouble();
    patternSensitivity = settings->value("consciousness/pattern_sensitivity", 0.7).toDouble();
    selfAwarenessLevel = settings->value("consciousness/self_awareness_level", 0.8).toDouble();

    // Load emotion settings
    emotionsEnabled = settings->value("emotions/enabled", true).toBool();
    maxResponseHistory = settings->value("emotions/max_response_history", 20).toInt();
    emotionsRepetitionPenalty = settings->value("emotions/repetition_penalty", 0.8).toDouble();
    enhanceResponses = settings->value("emotions/enhance_responses", true).toBool();

    // Load performance settings
    gpuUsageTarget = settings->value("performance/gpu_usage_target", 80).toInt();
    memoryUsageTarget = settings->value("performance/memory_usage_target", 85).toInt();
    temperatureThreshold = settings->value("performance/temperature_threshold", 80).toInt();
    enableMonitoring = settings->value("performance/enable_monitoring", true).toBool();
    optimizationInterval = settings->value("performance/optimization_interval", 60).toInt();

    // Load logging settings
    logLevel = settings->value("logging/level", "INFO").toString();
    logFile = settings->value("logging/file", "data/niodoo.log").toString();
    consoleLogLevel = settings->value("logging/console_level", "INFO").toString();
    enableStructuredLogging = settings->value("logging/enable_structured_logging", true).toBool();
    enableLogRotation = settings->value("logging/enable_log_rotation", true).toBool();
    maxLogFileSize = settings->value("logging/max_log_file_size", 10).toInt();

    qDebug() << "Configuration loaded successfully";
    qDebug() << "Emotion threshold:" << emotionThreshold;
    qDebug() << "Agents count:" << agentsCount;
    qDebug() << "Model:" << modelName;
}

void MainWindow::saveConfiguration()
{
    // Save core settings
    settings->setValue("core/emotion_threshold", emotionThreshold);
    settings->setValue("core/max_history", maxHistory);
    settings->setValue("core/db_path", dbPath);
    settings->setValue("core/backup_interval", backupInterval);
    settings->setValue("core/context_window", contextWindow);
    settings->setValue("core/response_delay", responseDelay);

    // Save model settings
    settings->setValue("models/default_model", modelName);
    settings->setValue("models/backup_model", backupModelName);
    settings->setValue("models/temperature", modelTemperature);
    settings->setValue("models/max_tokens", maxTokens);
    settings->setValue("models/timeout", modelTimeout);
    settings->setValue("models/top_p", topP);
    settings->setValue("models/top_k", topK);
    settings->setValue("models/repeat_penalty", repeatPenalty);
    settings->setValue("models/frequency_penalty", frequencyPenalty);
    settings->setValue("models/presence_penalty", presencePenalty);

    // Save Qt/GUI settings
    settings->setValue("qt/emotion_threshold", qtEmotionThreshold);
    settings->setValue("qt/distributed_mode", distributedMode);
    settings->setValue("qt/agents_count", agentsCount);
    settings->setValue("qt/connections_count", connectionsCount);
    settings->setValue("qt/architect_endpoint", architectEndpoint);
    settings->setValue("qt/developer_endpoint", developerEndpoint);
    settings->setValue("qt/hardware_acceleration", hardwareAcceleration);
    settings->setValue("qt/network_mode", networkMode);
    settings->setValue("qt/pathway_activation", pathwayActivation);

    // Save API settings
    settings->setValue("api/ollama_url", ollamaUrl);
    settings->setValue("api/api_timeout", apiTimeout);
    settings->setValue("api/retry_attempts", retryAttempts);
    settings->setValue("api/enable_caching", enableCaching);
    settings->setValue("api/cache_ttl", cacheTtl);

    // Save consciousness settings
    settings->setValue("consciousness/enabled", consciousnessEnabled);
    settings->setValue("consciousness/reflection_enabled", reflectionEnabled);
    settings->setValue("consciousness/emotion_sensitivity", emotionSensitivity);
    settings->setValue("consciousness/memory_threshold", memoryThreshold);
    settings->setValue("consciousness/pattern_sensitivity", patternSensitivity);
    settings->setValue("consciousness/self_awareness_level", selfAwarenessLevel);

    // Save emotion settings
    settings->setValue("emotions/enabled", emotionsEnabled);
    settings->setValue("emotions/max_response_history", maxResponseHistory);
    settings->setValue("emotions/repetition_penalty", emotionsRepetitionPenalty);
    settings->setValue("emotions/enhance_responses", enhanceResponses);

    // Save performance settings
    settings->setValue("performance/gpu_usage_target", gpuUsageTarget);
    settings->setValue("performance/memory_usage_target", memoryUsageTarget);
    settings->setValue("performance/temperature_threshold", temperatureThreshold);
    settings->setValue("performance/enable_monitoring", enableMonitoring);
    settings->setValue("performance/optimization_interval", optimizationInterval);

    // Save logging settings
    settings->setValue("logging/level", logLevel);
    settings->setValue("logging/file", logFile);
    settings->setValue("logging/console_level", consoleLogLevel);
    settings->setValue("logging/enable_structured_logging", enableStructuredLogging);
    settings->setValue("logging/enable_log_rotation", enableLogRotation);
    settings->setValue("logging/max_log_file_size", maxLogFileSize);

    settings->sync();
    qDebug() << "Configuration saved successfully";
}

MainWindow::~MainWindow()
{
    if (emotionalAI) delete emotionalAI;
    if (brainBridge) delete brainBridge;
    if (neuralEngine) delete neuralEngine;
    if (statusTimer) delete statusTimer;
    if (trayIcon) delete trayIcon;
    if (settings) delete settings;
}

void MainWindow::initializeComponents()
{
    // Initialize emotional AI manager with config values
    emotionalAI = new EmotionalAIManager();
    emotionalAI->setArchitectEndpoint(architectEndpoint);
    emotionalAI->setDeveloperEndpoint(developerEndpoint);
    emotionalAI->setDistributedMode(distributedMode);

    // Initialize brain system bridge with config values
    brainBridge = new BrainSystemBridge();
    brainBridge->setAgentsCount(agentsCount);
    brainBridge->setConnectionsCount(connectionsCount);

    // Initialize neural network engine with config values
    neuralEngine = new NeuralNetworkEngine();
    neuralEngine->setHardwareAcceleration(hardwareAcceleration);
#ifdef HAS_ONNXRUNTIME
    neuralEngine->initializeONNXRuntime();
    if (neuralEngine->isONNXInitialized()) {
        neuralEngine->loadEmotionalModel("models/emotion_detector.onnx");
        neuralEngine->loadConsensusModel("models/agent_consensus.onnx");
    } else {
        qDebug() << "ONNX not initialized, using mock inference";
    }
#else
    qDebug() << "ONNX not available, using mock inference";
#endif
}

void MainWindow::setupUI()
{
    centralWidget = new QWidget();
    setCentralWidget(centralWidget);

    // Create main layout
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

    // Create tab widget
    tabWidget = new QTabWidget();
    mainLayout->addWidget(tabWidget);

    // Setup individual tabs
    setupEmotionalTab();
    setupBrainTab();
    setupNeuralTab();
    setupHardwareTab();
}

void MainWindow::setupEmotionalTab()
{
    emotionalTab = new QWidget();
    tabWidget->addTab(emotionalTab, "Emotional Processing");

    QVBoxLayout* layout = new QVBoxLayout(emotionalTab);

    // Input section
    QGroupBox* inputGroup = new QGroupBox("Emotional Input");
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);

    emotionalInput = new QTextEdit();
    emotionalInput->setPlaceholderText("Describe your emotional state or situation...");
    emotionalInput->setMinimumHeight(100);
    inputLayout->addWidget(emotionalInput);

    processButton = new QPushButton("Process Emotions");
    processButton->setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #4CAF50; color: white; }");
    inputLayout->addWidget(processButton);

    processingProgress = new QProgressBar();
    processingProgress->setVisible(false);
    inputLayout->addWidget(processingProgress);

    layout->addWidget(inputGroup);

    // Output section
    QSplitter* splitter = new QSplitter(Qt::Horizontal);

    QGroupBox* outputGroup = new QGroupBox("Analysis Results");
    QVBoxLayout* outputLayout = new QVBoxLayout(outputGroup);
    emotionalOutput = new QTextEdit();
    emotionalOutput->setReadOnly(true);
    outputLayout->addWidget(emotionalOutput);
    splitter->addWidget(outputGroup);

    // Side panel for emotions and pathways
    QWidget* sidePanel = new QWidget();
    QVBoxLayout* sideLayout = new QVBoxLayout(sidePanel);

    detectedEmotions = new QListWidget();
    detectedEmotions->setMaximumWidth(250);
    sideLayout->addWidget(new QLabel("Detected Emotions:"));
    sideLayout->addWidget(detectedEmotions);

    activePathways = new QListWidget();
    activePathways->setMaximumWidth(250);
    sideLayout->addWidget(new QLabel("Active Neural Pathways:"));
    sideLayout->addWidget(activePathways);

    splitter->addWidget(sidePanel);
    splitter->setSizes(QList<int>() << 600 << 400);

    layout->addWidget(splitter);
}

void MainWindow::setupBrainTab()
{
    brainTab = new QWidget();
    tabWidget->addTab(brainTab, "Brain System");

    QVBoxLayout* layout = new QVBoxLayout(brainTab);

    // System stats
    QGroupBox* statsGroup = new QGroupBox("Brain System Statistics");
    QGridLayout* statsLayout = new QGridLayout(statsGroup);

    statsLayout->addWidget(new QLabel("Active Agents:"), 0, 0);
    agentsCountLabel = new QLabel(QString::number(agentsCount));
    statsLayout->addWidget(agentsCountLabel, 0, 1);

    statsLayout->addWidget(new QLabel("Neural Connections:"), 1, 0);
    connectionsCountLabel = new QLabel(QString::number(connectionsCount));
    statsLayout->addWidget(connectionsCountLabel, 1, 1);

    statsLayout->addWidget(new QLabel("Agent Consensus:"), 2, 0);
    consensusLabel = new QLabel("95.2%");
    statsLayout->addWidget(consensusLabel, 2, 1);

    layout->addWidget(statsGroup);

    // Activity visualization
    agentActivityBar = new QProgressBar();
    agentActivityBar->setRange(0, 100);
    agentActivityBar->setValue(75);
    agentActivityBar->setFormat("Agent Activity: %p%");
    layout->addWidget(agentActivityBar);

    // Status log
    brainStatusLog = new QTextEdit();
    brainStatusLog->setReadOnly(true);
    brainStatusLog->setMaximumHeight(200);
    layout->addWidget(new QLabel("Brain System Log:"));
    layout->addWidget(brainStatusLog);
}

void MainWindow::setupNeuralTab()
{
    neuralTab = new QWidget();
    tabWidget->addTab(neuralTab, "Neural Network");

    QVBoxLayout* layout = new QVBoxLayout(neuralTab);

    // Controls
    QGroupBox* controlsGroup = new QGroupBox("Neural Network Controls");
    QGridLayout* controlsLayout = new QGridLayout(controlsGroup);

    controlsLayout->addWidget(new QLabel("Pathway Activation:"), 0, 0);
    pathwayActivationSlider = new QSlider(Qt::Horizontal);
    pathwayActivationSlider->setRange(50, 200);
    pathwayActivationSlider->setValue(pathwayActivation);
    controlsLayout->addWidget(pathwayActivationSlider, 0, 1);

    controlsLayout->addWidget(new QLabel("Agent Count:"), 1, 0);
    agentsSpinBox = new QSpinBox();
    agentsSpinBox->setRange(50, 200);
    agentsSpinBox->setValue(agentsCount);
    controlsLayout->addWidget(agentsSpinBox, 1, 1);

    controlsLayout->addWidget(new QLabel("Hardware Acceleration:"), 2, 0);
    hardwareAccelCheckBox = new QCheckBox();
    hardwareAccelCheckBox->setChecked(hardwareAcceleration);
    controlsLayout->addWidget(hardwareAccelCheckBox, 2, 1);

    controlsLayout->addWidget(new QLabel("Network Mode:"), 3, 0);
    networkModeCombo = new QComboBox();
    networkModeCombo->addItems({"Local", "Distributed", "Hybrid"});
    networkModeCombo->setCurrentText(networkMode);
    controlsLayout->addWidget(networkModeCombo, 3, 1);

    layout->addWidget(controlsGroup);

    // Load visualization
    neuralLoadBar = new QProgressBar();
    neuralLoadBar->setRange(0, 100);
    neuralLoadBar->setValue(45);
    neuralLoadBar->setFormat("Neural Load: %p%");
    layout->addWidget(neuralLoadBar);

    // Metrics log
    neuralMetricsLog = new QTextEdit();
    neuralMetricsLog->setReadOnly(true);
    neuralMetricsLog->setMaximumHeight(200);
    layout->addWidget(new QLabel("Neural Metrics:"));
    layout->addWidget(neuralMetricsLog);
}

void MainWindow::setupHardwareTab()
{
    hardwareTab = new QWidget();
    tabWidget->addTab(hardwareTab, "Hardware");

    QVBoxLayout* layout = new QVBoxLayout(hardwareTab);

    // GPU and Memory usage
    QGroupBox* usageGroup = new QGroupBox("Hardware Usage");
    QGridLayout* usageLayout = new QGridLayout(usageGroup);

    usageLayout->addWidget(new QLabel("GPU Usage:"), 0, 0);
    gpuUsageLabel = new QLabel(QString("%1%").arg(gpuUsageTarget));
    usageLayout->addWidget(gpuUsageLabel, 0, 1);

    usageLayout->addWidget(new QLabel("Memory Usage:"), 1, 0);
    memoryUsageLabel = new QLabel(QString("%1 GB / 16 GB").arg(memoryUsageTarget * 16 / 100.0, 0, 'f', 1));
    usageLayout->addWidget(memoryUsageLabel, 1, 1);

    usageLayout->addWidget(new QLabel("Temperature:"), 2, 0);
    temperatureLabel = new QLabel(QString("%1°C").arg(temperatureThreshold));
    usageLayout->addWidget(temperatureLabel, 2, 1);

    layout->addWidget(usageGroup);

    // Progress bars
    gpuProgressBar = new QProgressBar();
    gpuProgressBar->setRange(0, 100);
    gpuProgressBar->setValue(gpuUsageTarget);
    gpuProgressBar->setFormat("GPU: %p%");
    layout->addWidget(gpuProgressBar);

    memoryProgressBar = new QProgressBar();
    memoryProgressBar->setRange(0, 100);
    memoryProgressBar->setValue(memoryUsageTarget);
    memoryProgressBar->setFormat("Memory: %p%");
    layout->addWidget(memoryProgressBar);

    // Controls
    optimizeButton = new QPushButton("Optimize Performance");
    optimizeButton->setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #2196F3; color: white; }");
    layout->addWidget(optimizeButton);
}

void MainWindow::connectSignals()
{
    connect(processButton, &QPushButton::clicked, this, &MainWindow::onProcessEmotionalInput);
    connect(pathwayActivationSlider, &QSlider::valueChanged, this, &MainWindow::onNeuralPathwayUpdate);
    connect(agentsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onAgentConsensusUpdate);
    connect(hardwareAccelCheckBox, &QCheckBox::toggled, this, &MainWindow::onHardwareAccelerationToggle);
    connect(networkModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onNetworkModeChanged);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::updateSystemStatus);
    connect(optimizeButton, &QPushButton::clicked, this, &MainWindow::onProcessEmotionalInput);
}

void MainWindow::onProcessEmotionalInput()
{
    QString input = emotionalInput->toPlainText().trimmed();
    if (input.isEmpty()) {
        QMessageBox::warning(this, "Input Required", "Please enter emotional input to process.");
        return;
    }

    processButton->setEnabled(false);
    processingProgress->setVisible(true);
    processingProgress->setRange(0, 0); // Indeterminate progress

    emotionalOutput->clear();
    detectedEmotions->clear();
    activePathways->clear();

    // Process through emotional AI system
    emotionalAI->processEmotionalInput(input);

    // Simulate processing (in real implementation, this would be async)
    QTimer::singleShot(2000, [this]() {
        processButton->setEnabled(true);
        processingProgress->setVisible(false);

        // Update UI with results
        emotionalOutput->setPlainText("Processing complete! Analysis shows complex emotional patterns with high empathy scores.");
        detectedEmotions->addItems({"Joy", "Anxiety", "Hope", "Fear"});
        activePathways->addItems({"Empathy Circuit", "Memory Integration", "Future Projection"});
    });
}

void MainWindow::onNeuralPathwayUpdate()
{
    int value = pathwayActivationSlider->value();
    brainBridge->setPathwayActivation(value / 100.0f);
    neuralMetricsLog->append(QString("Pathway activation updated to %1%").arg(value));
}

void MainWindow::onAgentConsensusUpdate()
{
    int agents = agentsSpinBox->value();
    brainBridge->setAgentsCount(agents);
    agentsCountLabel->setText(QString::number(agents));
    neuralMetricsLog->append(QString("Agent count updated to %1").arg(agents));
}

void MainWindow::onHardwareAccelerationToggle()
{
    bool enabled = hardwareAccelCheckBox->isChecked();
    neuralEngine->setHardwareAcceleration(enabled);
    neuralMetricsLog->append(QString("Hardware acceleration %1").arg(enabled ? "enabled" : "disabled"));
}

void MainWindow::onNetworkModeChanged()
{
    QString mode = networkModeCombo->currentText();
    distributedMode = (mode != "Local");
    emotionalAI->setDistributedMode(distributedMode);
    brainStatusLog->append(QString("Network mode changed to %1").arg(mode));
}

void MainWindow::updateSystemStatus()
{
    // Update system statistics
    agentsCountLabel->setText(QString::number(brainBridge->getAgentsCount()));
    connectionsCountLabel->setText(QString::number(brainBridge->getConnectionsCount()));
    consensusLabel->setText(QString("%1%").arg(brainBridge->getConsensusLevel(), 0, 'f', 1));

    agentActivityBar->setValue(brainBridge->getActivityLevel());

    // Update hardware stats
    gpuProgressBar->setValue(neuralEngine->getGPUUsage());
    memoryProgressBar->setValue(neuralEngine->getMemoryUsage());
    gpuUsageLabel->setText(QString("%1%").arg(neuralEngine->getGPUUsage()));
    memoryUsageLabel->setText(QString("%1 GB / 16 GB").arg(neuralEngine->getMemoryUsage() * 16 / 100.0, 0, 'f', 1));
    temperatureLabel->setText(QString("%1°C").arg(neuralEngine->getTemperature()));

    // Update status bar
    statusBar()->showMessage(QString("Brain System Active - Agents: %1, Consensus: %2%")
                            .arg(brainBridge->getAgentsCount())
                            .arg(brainBridge->getConsensusLevel(), 0, 'f', 1));
}

void MainWindow::setupTrayIcon()
{
    if (!QSystemTrayIcon::isSystemTrayAvailable()) {
        return;
    }

    trayIcon = new QSystemTrayIcon(this);
    trayIcon->setIcon(QIcon(":/icons/brain.png"));

    QMenu* trayMenu = new QMenu();
    trayMenu->addAction("Show", this, &MainWindow::show);
    trayMenu->addAction("Process Emotions", this, &MainWindow::onProcessEmotionalInput);
    trayMenu->addSeparator();
    trayMenu->addAction("Quit", this, &QWidget::close);

    trayIcon->setContextMenu(trayMenu);
    trayIcon->show();

    connect(trayIcon, &QSystemTrayIcon::activated, [this](QSystemTrayIcon::ActivationReason reason) {
        if (reason == QSystemTrayIcon::DoubleClick) {
            show();
            raise();
            activateWindow();
        }
    });
}

void MainWindow::showAbout()
{
    QMessageBox::about(this, "Brain Integration System",
        QString("<h2>Brain Integration - Emotional AI</h2>"
        "<p>Advanced emotional intelligence system with %1 neural agents</p>"
        "<p>Distributed consciousness architecture</p>"
        "<p>Model: %2 | Temperature: %3</p>"
        "<p>Built with Qt6 and ONNX Runtime</p>").arg(agentsCount).arg(modelName).arg(modelTemperature));
}


void MainWindow::closeEvent(QCloseEvent* event)
{
    if (trayIcon && trayIcon->isVisible()) {
        hide();
        event->ignore();
        return;
    }
    event->accept();
}

void MainWindow::setupMenus()
{
    // Create menu bar
    QMenuBar* menuBar = this->menuBar();
    
    // File menu
    QMenu* fileMenu = menuBar->addMenu("&File");
    fileMenu->addAction("&Process Emotions", this, &MainWindow::onProcessEmotionalInput, QKeySequence::New);
    fileMenu->addSeparator();
    fileMenu->addAction("&Exit", this, &QWidget::close, QKeySequence::Quit);
    
    // View menu
    QMenu* viewMenu = menuBar->addMenu("&View");
    viewMenu->addAction("&Emotional Processing", [this]() { tabWidget->setCurrentIndex(0); });
    viewMenu->addAction("&Brain Statistics", [this]() { tabWidget->setCurrentIndex(1); });
    viewMenu->addAction("&Neural Control", [this]() { tabWidget->setCurrentIndex(2); });
    viewMenu->addAction("&Hardware Metrics", [this]() { tabWidget->setCurrentIndex(3); });
    
    // Help menu
    QMenu* helpMenu = menuBar->addMenu("&Help");
    helpMenu->addAction("&About", this, &MainWindow::showAbout);
    helpMenu->addAction("About &Qt", qApp, &QApplication::aboutQt);
}

void MainWindow::setupStatusBar()
{
    // Create status bar
    QStatusBar* statusBar = this->statusBar();
    
    // Add permanent widgets to status bar
    QLabel* connectionStatus = new QLabel("Connected");
    connectionStatus->setStyleSheet("QLabel { color: green; }");
    statusBar->addPermanentWidget(connectionStatus);
    
    QLabel* agentStatus = new QLabel(QString("Agents: %1").arg(agentsCount));
    statusBar->addPermanentWidget(agentStatus);
    
    QLabel* gpuStatus = new QLabel("GPU: Ready");
    statusBar->addPermanentWidget(gpuStatus);
    
    // Set initial message
    statusBar->showMessage("Brain Integration System Ready", 3000);
}

// #include "MainWindow.moc" // Generated by Qt MOC
