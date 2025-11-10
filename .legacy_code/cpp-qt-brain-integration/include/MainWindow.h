#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QTextEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QListWidget>
#include <QTabWidget>
#include <QStatusBar>
#include <QMenuBar>
#include <QGroupBox>
#include <QSpinBox>
#include <QSlider>
#include <QCheckBox>
#include <QComboBox>
#include <QSplitter>
#include <QScrollArea>
#include <QTimer>
#include <QSystemTrayIcon>
#include <QSettings>

#include "EmotionalAIManager.h"
#include "BrainSystemBridge.h"
#include "NeuralNetworkEngine.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onProcessEmotionalInput();
    void onNeuralPathwayUpdate();
    void onAgentConsensusUpdate();
    void onHardwareAccelerationToggle();
    void onNetworkModeChanged();
    void updateSystemStatus();
    void showAbout();

private:
    void setupUI();
    void setupMenus();
    void setupStatusBar();
    void connectSignals();
    void initializeComponents();
    void setupEmotionalTab();
    void setupBrainTab();
    void setupNeuralTab();
    void setupHardwareTab();
    void setupTrayIcon();

    // Core systems
    EmotionalAIManager* emotionalAI;
    BrainSystemBridge* brainBridge;
    NeuralNetworkEngine* neuralEngine;

    // UI Components
    QWidget* centralWidget;
    QTabWidget* tabWidget;

    // Emotional Processing Tab
    QWidget* emotionalTab;
    QTextEdit* emotionalInput;
    QTextEdit* emotionalOutput;
    QPushButton* processButton;
    QProgressBar* processingProgress;
    QListWidget* detectedEmotions;
    QListWidget* activePathways;

    // Brain System Tab
    QWidget* brainTab;
    QLabel* agentsCountLabel;
    QLabel* connectionsCountLabel;
    QLabel* consensusLabel;
    QProgressBar* agentActivityBar;
    QTextEdit* brainStatusLog;

    // Neural Network Tab
    QWidget* neuralTab;
    QSlider* pathwayActivationSlider;
    QSpinBox* agentsSpinBox;
    QCheckBox* hardwareAccelCheckBox;
    QComboBox* networkModeCombo;
    QProgressBar* neuralLoadBar;
    QTextEdit* neuralMetricsLog;

    // Hardware Tab
    QWidget* hardwareTab;
    QLabel* gpuUsageLabel;
    QLabel* memoryUsageLabel;
    QLabel* temperatureLabel;
    QProgressBar* gpuProgressBar;
    QProgressBar* memoryProgressBar;
    QPushButton* optimizeButton;

    // System monitoring
    QTimer* statusTimer;
    QSystemTrayIcon* trayIcon;

    // Configuration system
    QSettings* settings;
    void loadConfiguration();
    void saveConfiguration();

    // Core configuration variables
    double emotionThreshold;
    int maxHistory;
    QString dbPath;
    int backupInterval;
    int contextWindow;
    double responseDelay;

    // Model configuration variables
    QString modelName;
    QString backupModelName;
    double modelTemperature;
    int maxTokens;
    int modelTimeout;
    double topP;
    int topK;
    double repeatPenalty;
    double frequencyPenalty;
    double presencePenalty;

    // Qt/GUI configuration variables
    double qtEmotionThreshold;
    bool distributedMode;
    int agentsCount;
    int connectionsCount;
    QString architectEndpoint;
    QString developerEndpoint;
    bool hardwareAcceleration;
    QString networkMode;
    int pathwayActivation;

    // API configuration variables
    QString ollamaUrl;
    int apiTimeout;
    int retryAttempts;
    bool enableCaching;
    int cacheTtl;

    // Consciousness configuration variables
    bool consciousnessEnabled;
    bool reflectionEnabled;
    double emotionSensitivity;
    double memoryThreshold;
    double patternSensitivity;
    double selfAwarenessLevel;

    // Emotion configuration variables
    bool emotionsEnabled;
    int maxResponseHistory;
    double emotionsRepetitionPenalty;
    bool enhanceResponses;

    // Performance configuration variables
    int gpuUsageTarget;
    int memoryUsageTarget;
    int temperatureThreshold;
    bool enableMonitoring;
    int optimizationInterval;

    // Logging configuration variables
    QString logLevel;
    QString logFile;
    QString consoleLogLevel;
    bool enableStructuredLogging;
    bool enableLogRotation;
    int maxLogFileSize;

protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // MAINWINDOW_H
