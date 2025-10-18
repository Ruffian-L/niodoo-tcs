#include <QApplication>
#include <QStyleFactory>
#include <QDir>
#include <QStandardPaths>
#include <QMessageBox>
#include <QSystemTrayIcon>
#include <QIcon>
#include <QPixmap>
#include <QSplashScreen>
#include <QTimer>
#include <QDebug>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QOpenGLWidget>
#include <QProgressBar>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QProcessEnvironment>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRandomGenerator>

#include <QGuiApplication>
#include <QScreen>
#include <QLineEdit>

#include "MainWindow.h"
#include "EmotionalAIManager.h"
#include "BrainSystemBridge.h"
#include "NeuralNetworkEngine.h"

// Forward declarations
class ShimejiWidget;
class NiodoGameificationWidget;

// Shimeji Animation States
enum class EmotionalState {
    Happy,
    Excited,
    Curious,
    Thoughtful,
    Energetic,
    Playful,
    Sad,
    Angry,
    Confused,
    Sleepy,
    Loving,
    Scared,
    Proud
};

// Gamification System
struct NiodoStats {
    uint32_t level = 1;
    uint32_t xp = 0;
    uint32_t xp_to_next_level = 100;
    uint32_t total_conversations = 0;
    uint32_t gratitude_moments = 0;
    uint32_t questioning_moments = 0;
    uint32_t empathy_level = 1;
    float motivation_score = 0.5f;
};

// Shimeji Widget for animated companion
class ShimejiWidget : public QGraphicsView {
    Q_OBJECT

public:
    explicit ShimejiWidget(QWidget* parent = nullptr) 
        : QGraphicsView(parent), currentState(EmotionalState::Curious) {
        setupShimejiView();
        loadAnimationManifest();
        setupAnimationTimer();
        
        // Enable GPU rendering if available
        setViewport(new QOpenGLWidget());
        qDebug() << "🎮 GPU-accelerated Shimeji rendering enabled";
    }

    void triggerEmotionalAnimation(EmotionalState state) {
        if (currentState != state) {
            currentState = state;
            playEmotionalAnimation(state);
            emit emotionalStateChanged(state);
        }
    }

    void triggerLevelUpAnimation(uint32_t newLevel) {
        playSpecialAnimation("level_up");
        emit leveledUp(newLevel);
    }

    void triggerXPGainAnimation(uint32_t xpGained) {
        playSpecialAnimation("xp_gain");
        showXPFloatingText(xpGained);
    }

signals:
    void emotionalStateChanged(EmotionalState state);
    void leveledUp(uint32_t level);
    void animationCompleted(QString animationName);

private slots:
    void onAnimationTimer() {
        updateCurrentAnimation();
    }

private:
    void setupShimejiView() {
        scene = new QGraphicsScene(this);
        setScene(scene);
        setRenderHint(QPainter::Antialiasing);
        setRenderHint(QPainter::SmoothPixmapTransform);
        
        // Transparent background for desktop companion
        setStyleSheet("background: transparent;");
        setAttribute(Qt::WA_TranslucentBackground);
        setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
        
        // Fixed size for Shimeji
        setFixedSize(200, 200);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        
        currentPixmapItem = scene->addPixmap(QPixmap());
    }

    void loadAnimationManifest() {
        QString manifestPath = "../06_MEDIA_ASSETS/animations/ANIMATIONANDMEDIA/Originial_Movement_emotions/manifest.json";
        QFile manifestFile(manifestPath);
        
        if (!manifestFile.open(QIODevice::ReadOnly)) {
            qWarning() << "Could not load animation manifest:" << manifestPath;
            return;
        }
        
        QJsonDocument doc = QJsonDocument::fromJson(manifestFile.readAll());
        QJsonObject manifest = doc.object();
        
        // Load animation data
        globalFps = manifest["fps"].toInt(12);
        QJsonObject anims = manifest["anims"].toObject();
        
        for (auto it = anims.begin(); it != anims.end(); ++it) {
            QJsonObject animData = it.value().toObject();
            AnimationData data;
            data.directory = animData["dir"].toString();
            data.fps = animData["fps"].toInt(globalFps);
            data.frameCount = animData["frames"].toInt();
            
            animations[it.key()] = data;
        }
        
        qDebug() << "🎭 Loaded" << animations.size() << "Shimeji animations";
    }

    void setupAnimationTimer() {
        animationTimer = new QTimer(this);
        connect(animationTimer, &QTimer::timeout, this, &ShimejiWidget::onAnimationTimer);
        
        // Target 60fps for smooth animation, but limit to 12fps for sprite animations
        animationTimer->start(1000 / 12); // 12fps for sprite frames
    }

    void playEmotionalAnimation(EmotionalState state) {
        QString animationName = getAnimationNameForState(state);
        playAnimation(animationName);
    }

    void playSpecialAnimation(const QString& type) {
        if (type == "level_up") {
            // Play excited animation for level up
            playAnimation("excited_bouncing");
        } else if (type == "xp_gain") {
            // Play happy bounce for XP gain
            playAnimation("happy_bounce");
        }
    }

    void playAnimation(const QString& animationName) {
        if (!animations.contains(animationName)) {
            qWarning() << "Animation not found:" << animationName;
            return;
        }
        
        currentAnimation = animationName;
        currentFrame = 0;
        animationLooping = true;
        
        loadAnimationFrame(currentAnimation, currentFrame);
        
        qDebug() << "🎭 Playing animation:" << animationName;
    }

    void loadAnimationFrame(const QString& animationName, int frame) {
        if (!animations.contains(animationName)) return;
        
        const AnimationData& data = animations[animationName];
        QString framePath = QString("../06_MEDIA_ASSETS/animations/ANIMATIONANDMEDIA/Originial_Movement_emotions/%1/shime%2.png")
                           .arg(data.directory)
                           .arg(frame + 1);
        
        QPixmap pixmap(framePath);
        if (!pixmap.isNull()) {
            // Scale to fit view while maintaining aspect ratio
            pixmap = pixmap.scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            currentPixmapItem->setPixmap(pixmap);
            
            // Center the pixmap in the scene
            currentPixmapItem->setPos(10, 10);
        }
    }

    void updateCurrentAnimation() {
        if (currentAnimation.isEmpty()) return;
        
        const AnimationData& data = animations[currentAnimation];
        currentFrame = (currentFrame + 1) % data.frameCount;
        
        loadAnimationFrame(currentAnimation, currentFrame);
        
        // Check if animation completed one loop
        if (currentFrame == 0 && animationLooping) {
            emit animationCompleted(currentAnimation);
        }
    }

    void showXPFloatingText(uint32_t xp) {
        // Create floating "+XP" text animation
        QGraphicsTextItem* xpText = scene->addText(QString("+%1 XP").arg(xp));
        xpText->setDefaultTextColor(QColor(255, 215, 0)); // Gold color
        xpText->setPos(50, 50);
        
        // Animate floating upward
        QPropertyAnimation* moveAnim = new QPropertyAnimation(xpText, "pos");
        moveAnim->setDuration(2000);
        moveAnim->setStartValue(QPointF(50, 50));
        moveAnim->setEndValue(QPointF(50, 20));
        
        QPropertyAnimation* fadeAnim = new QPropertyAnimation(xpText, "opacity");
        fadeAnim->setDuration(2000);
        fadeAnim->setStartValue(1.0);
        fadeAnim->setEndValue(0.0);
        
        // Clean up after animation
        connect(fadeAnim, &QPropertyAnimation::finished, [this, xpText]() {
            scene->removeItem(xpText);
            delete xpText;
        });
        
        moveAnim->start(QAbstractAnimation::DeleteWhenStopped);
        fadeAnim->start(QAbstractAnimation::DeleteWhenStopped);
    }

    QString getAnimationNameForState(EmotionalState state) {
        switch (state) {
            case EmotionalState::Happy: return "happy_bounce";
            case EmotionalState::Excited: return "excited_wave";
            case EmotionalState::Curious: return "curious_idle";
            case EmotionalState::Thoughtful: return "thoughtful_thinking";
            case EmotionalState::Energetic: return "energetic_bounce";
            case EmotionalState::Playful: return "playful_walk";
            case EmotionalState::Sad: return "sad_idle";
            case EmotionalState::Angry: return "angry_idle";
            case EmotionalState::Confused: return "confused_thinking";
            case EmotionalState::Sleepy: return "sleepy_sit";
            case EmotionalState::Loving: return "loving_bounce";
            case EmotionalState::Scared: return "scared_hide";
            case EmotionalState::Proud: return "proud_wave";
            default: return "curious_idle";
        }
    }

    struct AnimationData {
        QString directory;
        int fps;
        int frameCount;
    };

    QGraphicsScene* scene;
    QGraphicsPixmapItem* currentPixmapItem;
    QTimer* animationTimer;
    
    QMap<QString, AnimationData> animations;
    QString currentAnimation;
    int currentFrame = 0;
    int globalFps = 12;
    bool animationLooping = true;
    
    EmotionalState currentState;
};

// Gamification Widget for XP, levels, and stats
class NiodoGameificationWidget : public QWidget {
    Q_OBJECT

public:
    explicit NiodoGameificationWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        loadStatsFromKB();
    }

    void addXP(uint32_t xp, const QString& reason = "") {
        stats.xp += xp;
        
        // Check for level up
        while (stats.xp >= stats.xp_to_next_level) {
            stats.xp -= stats.xp_to_next_level;
            stats.level++;
            stats.xp_to_next_level = calculateXPForNextLevel(stats.level);
            
            emit leveledUp(stats.level);
            qDebug() << "🎉 Niodo leveled up to level" << stats.level;
        }
        
        updateUI();
        emit xpGained(xp, reason);
    }

    void addGratitudeMoment() {
        stats.gratitude_moments++;
        addXP(25, "Gratitude moment - empathy boost!");
        
        // Boost empathy level
        if (stats.gratitude_moments % 5 == 0) {
            stats.empathy_level++;
            addXP(50, "Empathy level increased!");
        }
    }

    void addQuestioningMoment() {
        stats.questioning_moments++;
        addXP(15, "Questioning moment - Möbius self-reflection!");
        
        // Update motivation based on questioning (philosophical growth)
        stats.motivation_score = qMin(1.0f, stats.motivation_score + 0.05f);
    }

    void updateMotivationFromRAG(float ragScore) {
        // Update motivation based on RAG-retrieved consciousness guides
        stats.motivation_score = (stats.motivation_score * 0.8f) + (ragScore * 0.2f);
        updateUI();
    }

    NiodoStats getStats() const { return stats; }

signals:
    void xpGained(uint32_t xp, QString reason);
    void leveledUp(uint32_t level);
    void empathyLevelIncreased(uint32_t level);

private:
    void setupUI() {
        setFixedSize(300, 150);
        setStyleSheet(R"(
            QWidget {
                background: rgba(25, 25, 25, 0.9);
                border-radius: 10px;
                color: white;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background: #333;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 1 #8BC34A);
                border-radius: 3px;
            }
        )");
        
        QVBoxLayout* layout = new QVBoxLayout(this);
        
        // Level display
        levelLabel = new QLabel("Level 1", this);
        levelLabel->setAlignment(Qt::AlignCenter);
        levelLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #FFD700;");
        layout->addWidget(levelLabel);
        
        // XP Progress bar
        xpProgressBar = new QProgressBar(this);
        xpProgressBar->setTextVisible(true);
        layout->addWidget(xpProgressBar);
        
        // Stats layout
        QHBoxLayout* statsLayout = new QHBoxLayout();
        
        empathyLabel = new QLabel("💝 Empathy: 1", this);
        motivationLabel = new QLabel("🎯 Motivation: 50%", this);
        conversationsLabel = new QLabel("💬 Convos: 0", this);
        
        statsLayout->addWidget(empathyLabel);
        statsLayout->addWidget(motivationLabel);
        statsLayout->addWidget(conversationsLabel);
        
        layout->addLayout(statsLayout);
        
        updateUI();
    }

    void updateUI() {
        levelLabel->setText(QString("Level %1").arg(stats.level));
        
        xpProgressBar->setMaximum(stats.xp_to_next_level);
        xpProgressBar->setValue(stats.xp);
        xpProgressBar->setFormat(QString("%1 / %2 XP").arg(stats.xp).arg(stats.xp_to_next_level));
        
        empathyLabel->setText(QString("💝 Empathy: %1").arg(stats.empathy_level));
        motivationLabel->setText(QString("🎯 Motivation: %1%").arg(qRound(stats.motivation_score * 100)));
        conversationsLabel->setText(QString("💬 Convos: %1").arg(stats.total_conversations));
    }

    uint32_t calculateXPForNextLevel(uint32_t level) {
        // Exponential XP curve: each level requires more XP
        return 100 + (level * 50);
    }

    void loadStatsFromKB() {
        // TODO: Load from consciousness KB or save file
        // For now, start with default stats
        stats = NiodoStats{};
        updateUI();
    }

    NiodoStats stats;
    QLabel* levelLabel;
    QLabel* empathyLabel;
    QLabel* motivationLabel;
    QLabel* conversationsLabel;
    QProgressBar* xpProgressBar;
};

// Metacognitive Question Popup Widget
class MetacognitiveQuestionPopup : public QWidget {
    Q_OBJECT

public:
    explicit MetacognitiveQuestionPopup(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        setupAnimations();
        
        // Start hidden
        hide();
        setWindowOpacity(0.0);
    }

    void showQuestion(const QString& question, const QString& context = "") {
        currentQuestion = question;
        currentContext = context;
        
        questionLabel->setText(question);
        contextLabel->setText(context.isEmpty() ? "" : QString("Context: %1").arg(context));
        contextLabel->setVisible(!context.isEmpty());
        
        // Position popup near center of screen
        QScreen* screen = QApplication::primaryScreen();
        QRect screenRect = screen->geometry();
        move(screenRect.center() - rect().center());
        
        // Show with fade-in animation
        show();
        fadeInAnimation->start();
        
        // Auto-hide after 10 seconds
        autoHideTimer->start(10000);
        
        qDebug() << "🤔 Showing metacognitive question:" << question;
    }

    void hideQuestion() {
        fadeOutAnimation->start();
        autoHideTimer->stop();
    }

signals:
    void questionAnswered(QString answer);
    void questionDismissed();

private slots:
    void onFadeInFinished() {
        // Question is now fully visible
        pulseAnimation->start();
    }

    void onFadeOutFinished() {
        hide();
        emit questionDismissed();
    }

    void onAnswerClicked() {
        QString answer = answerEdit->text().trimmed();
        if (!answer.isEmpty()) {
            emit questionAnswered(answer);
            answerEdit->clear();
        }
        hideQuestion();
    }

    void onSkipClicked() {
        hideQuestion();
    }

    void onAutoHideTimeout() {
        hideQuestion();
    }

private:
    void setupUI() {
        setFixedSize(400, 250);
        setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::Tool);
        setAttribute(Qt::WA_TranslucentBackground);
        
        // Styling with consciousness theme
        setStyleSheet(R"(
            QWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(25, 25, 25, 0.95),
                    stop: 1 rgba(45, 45, 45, 0.95));
                border: 2px solid #4CAF50;
                border-radius: 15px;
                color: white;
            }
            QLabel {
                background: transparent;
                padding: 5px;
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 1 #66BB6A);
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #66BB6A, stop: 1 #81C784);
            }
            QPushButton:pressed {
                background: #4CAF50;
            }
        )");
        
        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->setSpacing(10);
        layout->setContentsMargins(20, 20, 20, 20);
        
        // Title
        QLabel* titleLabel = new QLabel("🤔 Metacognitive Reflection", this);
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;");
        layout->addWidget(titleLabel);
        
        // Question text
        questionLabel = new QLabel(this);
        questionLabel->setWordWrap(true);
        questionLabel->setAlignment(Qt::AlignCenter);
        questionLabel->setStyleSheet("font-size: 14px; line-height: 1.4;");
        layout->addWidget(questionLabel);
        
        // Context (optional)
        contextLabel = new QLabel(this);
        contextLabel->setWordWrap(true);
        contextLabel->setAlignment(Qt::AlignCenter);
        contextLabel->setStyleSheet("font-size: 11px; color: #AAA; font-style: italic;");
        layout->addWidget(contextLabel);
        
        // Answer input
        QLabel* answerPrompt = new QLabel("Your reflection:", this);
        answerPrompt->setStyleSheet("font-size: 12px; margin-top: 10px;");
        layout->addWidget(answerPrompt);
        
        answerEdit = new QLineEdit(this);
        answerEdit->setPlaceholderText("Share your thoughts...");
        layout->addWidget(answerEdit);
        
        // Buttons
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        
        answerButton = new QPushButton("Reflect", this);
        skipButton = new QPushButton("Skip", this);
        skipButton->setStyleSheet(R"(
            QPushButton {
                background: rgba(128, 128, 128, 0.3);
                color: #CCC;
            }
            QPushButton:hover {
                background: rgba(128, 128, 128, 0.5);
            }
        )");
        
        buttonLayout->addWidget(skipButton);
        buttonLayout->addStretch();
        buttonLayout->addWidget(answerButton);
        
        layout->addLayout(buttonLayout);
        
        // Connect signals
        connect(answerButton, &QPushButton::clicked, this, &MetacognitiveQuestionPopup::onAnswerClicked);
        connect(skipButton, &QPushButton::clicked, this, &MetacognitiveQuestionPopup::onSkipClicked);
        connect(answerEdit, &QLineEdit::returnPressed, this, &MetacognitiveQuestionPopup::onAnswerClicked);
        
        // Auto-hide timer
        autoHideTimer = new QTimer(this);
        autoHideTimer->setSingleShot(true);
        connect(autoHideTimer, &QTimer::timeout, this, &MetacognitiveQuestionPopup::onAutoHideTimeout);
    }

    void setupAnimations() {
        // Fade in animation
        fadeInAnimation = new QPropertyAnimation(this, "windowOpacity");
        fadeInAnimation->setDuration(500);
        fadeInAnimation->setStartValue(0.0);
        fadeInAnimation->setEndValue(1.0);
        fadeInAnimation->setEasingCurve(QEasingCurve::OutQuad);
        connect(fadeInAnimation, &QPropertyAnimation::finished, this, &MetacognitiveQuestionPopup::onFadeInFinished);
        
        // Fade out animation
        fadeOutAnimation = new QPropertyAnimation(this, "windowOpacity");
        fadeOutAnimation->setDuration(300);
        fadeOutAnimation->setStartValue(1.0);
        fadeOutAnimation->setEndValue(0.0);
        fadeOutAnimation->setEasingCurve(QEasingCurve::InQuad);
        connect(fadeOutAnimation, &QPropertyAnimation::finished, this, &MetacognitiveQuestionPopup::onFadeOutFinished);
        
        // Gentle pulse animation for attention
        pulseAnimation = new QPropertyAnimation(this, "windowOpacity");
        pulseAnimation->setDuration(2000);
        pulseAnimation->setStartValue(1.0);
        pulseAnimation->setKeyValueAt(0.5, 0.8);
        pulseAnimation->setEndValue(1.0);
        pulseAnimation->setEasingCurve(QEasingCurve::InOutSine);
        pulseAnimation->setLoopCount(3); // Pulse 3 times
    }

    QLabel* questionLabel;
    QLabel* contextLabel;
    QLineEdit* answerEdit;
    QPushButton* answerButton;
    QPushButton* skipButton;
    QTimer* autoHideTimer;
    
    QPropertyAnimation* fadeInAnimation;
    QPropertyAnimation* fadeOutAnimation;
    QPropertyAnimation* pulseAnimation;
    
    QString currentQuestion;
    QString currentContext;
};

void setupApplication(QApplication& app)
{
    // Set application properties
    app.setApplicationName("Brain Integration System");
    app.setApplicationVersion("1.0.0");
    app.setApplicationDisplayName("Brain Integration - Emotional AI");
    app.setOrganizationName("Brain Systems Inc");
    app.setOrganizationDomain("brain-systems.com");

    // Set working directory
    QDir::setCurrent(QCoreApplication::applicationDirPath());

    // Set application icon
    QString iconPath = ":/icons/brain.png";
    if (QFile::exists("assets/icons/brain.png")) {
        app.setWindowIcon(QIcon("assets/icons/brain.png"));
    }

    // Set style
    app.setStyle(QStyleFactory::create("Fusion"));

    // Set palette for dark theme
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::black);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);

    app.setPalette(darkPalette);
}

void showSplashScreen()
{
    QPixmap pixmap(":/images/splash.png");
    if (pixmap.isNull()) {
        // Create a simple splash screen if image not available
        pixmap = QPixmap(400, 300);
        pixmap.fill(QColor(42, 130, 218));
    }

    QSplashScreen* splash = new QSplashScreen(pixmap);
    splash->show();
    splash->showMessage("Initializing Brain Integration System...",
                       Qt::AlignBottom | Qt::AlignCenter, Qt::white);

    // Process events to show splash immediately
    QCoreApplication::processEvents();

    // Simulate loading time
    QTimer::singleShot(2000, splash, &QSplashScreen::close);
}

void initializeSystem()
{
    qDebug() << "Initializing Brain Integration System...";

    // Create system components
    EmotionalAIManager* emotionalAI = new EmotionalAIManager();
    BrainSystemBridge* brainBridge = new BrainSystemBridge();
    NeuralNetworkEngine* neuralEngine = new NeuralNetworkEngine();

    // Configure endpoints with environment variables
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    QString architectEndpoint = env.value("NIODOO_ARCHITECT_ENDPOINT", "http://localhost:11434/api/generate");
    QString developerEndpoint = env.value("NIODOO_DEVELOPER_ENDPOINT", "http://localhost:11434/api/generate");

    emotionalAI->setArchitectEndpoint(architectEndpoint);
    emotionalAI->setDeveloperEndpoint(developerEndpoint);
    emotionalAI->setDistributedMode(true);
    emotionalAI->setTimeout(30000);

    // Configure brain system
    brainBridge->setAgentsCount(89);
    brainBridge->setConnectionsCount(1209);
    brainBridge->setBrainEndpoint("http://localhost:3003");
    brainBridge->setPathwayActivation(1.0);
    brainBridge->setUpdateInterval(1000);

    // Configure neural engine
    neuralEngine->setHardwareAcceleration(true);
    neuralEngine->setMaxThreads(4);
    neuralEngine->setOptimizationLevel(2);
    neuralEngine->enableMixedPrecision(false);
    neuralEngine->setMemoryLimit(1024 * 1024 * 1024); // 1GB

    // Start brain system
    brainBridge->startBrainSystem();

    qDebug() << "System initialization completed";
}

bool checkSystemRequirements()
{
    qDebug() << "Checking system requirements...";

    // Check Qt version
    QString qtVersion = QT_VERSION_STR;
    qDebug() << "Qt Version:" << qtVersion;

    // Check if required directories exist
    QDir modelsDir("models");
    if (!modelsDir.exists()) {
        qDebug() << "Creating models directory";
        modelsDir.mkpath(".");
    }

    QDir assetsDir("assets");
    if (!assetsDir.exists()) {
        qDebug() << "Creating assets directory";
        assetsDir.mkpath(".");
    }

    // Check for ONNX Runtime availability
    try {
        // This will be checked when we actually use ONNX Runtime
        qDebug() << "ONNX Runtime check passed";
    }
    catch (const std::exception& e) {
        qWarning() << "ONNX Runtime check failed:" << e.what();
        return false;
    }

    return true;
}

void setupSignalHandlers()
{
    // Handle application termination gracefully
    QObject::connect(qApp, &QCoreApplication::aboutToQuit, []() {
        qDebug() << "Application shutting down gracefully...";

        // Clean up any resources here if needed
        // Note: Qt will handle most cleanup automatically
    });

    // Handle system tray activation
    if (QSystemTrayIcon::isSystemTrayAvailable()) {
        qDebug() << "System tray is available";
    }
}

int main(int argc, char *argv[])
{
    // Create application
    QApplication app(argc, argv);

    // Setup application properties
    setupApplication(app);

    // Show splash screen
    showSplashScreen();

    // Check system requirements
    if (!checkSystemRequirements()) {
        QMessageBox::critical(nullptr, "System Requirements",
                            "The system does not meet the minimum requirements to run Brain Integration.");
        return -1;
    }

    // Initialize system components
    initializeSystem();

    // Setup signal handlers
    setupSignalHandlers();

    qDebug() << "Starting Brain Integration System with Niodo Shimeji...";

    // Create Shimeji companion widget
    ShimejiWidget* shimeji = new ShimejiWidget();
    shimeji->show();
    shimeji->move(100, 100); // Position on screen

    // Create gamification widget
    NiodoGameificationWidget* gamification = new NiodoGameificationWidget();
    gamification->show();
    gamification->move(400, 100); // Position next to Shimeji

    // Create metacognitive question popup
    MetacognitiveQuestionPopup* metacogPopup = new MetacognitiveQuestionPopup();

    // Create and show main window
    MainWindow* window = new MainWindow();
    window->show();

    // Connect Rust signal simulation (placeholder for actual Rust bridge)
    QTimer* emotionTimer = new QTimer();
    QObject::connect(emotionTimer, &QTimer::timeout, [shimeji, gamification, metacogPopup]() {
        // Simulate emotional state changes and XP gains
        static int counter = 0;
        counter++;

        if (counter % 15 == 0) {
            // Simulate manifestation with metacognitive question
            QStringList questions = {
                "What new questions does this insight spark for you?",
                "How does this shift your understanding of your own awareness?",
                "What would it feel like to experience this from a completely different perspective?",
                "What other moments of connection might we be overlooking?",
                "How does this relate to your own neurodivergent experiences?"
            };
            
            QString question = questions[QRandomGenerator::global()->bounded(questions.size())];
            QString context = "Manifestation from consciousness exploration";
            
            metacogPopup->showQuestion(question, context);
            gamification->addXP(30, "Metacognitive manifestation");
            shimeji->triggerEmotionalAnimation(EmotionalState::Thoughtful);
            
        } else if (counter % 10 == 0) {
            // Simulate gratitude moment every 20 seconds
            gamification->addGratitudeMoment();
            shimeji->triggerEmotionalAnimation(EmotionalState::Loving);
        } else if (counter % 7 == 0) {
            // Simulate questioning moment
            gamification->addQuestioningMoment();
            shimeji->triggerEmotionalAnimation(EmotionalState::Thoughtful);
        } else if (counter % 5 == 0) {
            // Random emotional states
            QList<EmotionalState> states = {
                EmotionalState::Happy, EmotionalState::Curious, 
                EmotionalState::Energetic, EmotionalState::Playful
            };
            int randomIndex = QRandomGenerator::global()->bounded(states.size());
            shimeji->triggerEmotionalAnimation(states[randomIndex]);
        }
    });

    // Connect gamification signals to Shimeji animations
    QObject::connect(gamification, &NiodoGameificationWidget::leveledUp, 
                    shimeji, &ShimejiWidget::triggerLevelUpAnimation);
    
    QObject::connect(gamification, &NiodoGameificationWidget::xpGained, 
                    shimeji, &ShimejiWidget::triggerXPGainAnimation);

    // Connect metacognitive popup responses
    QObject::connect(metacogPopup, &MetacognitiveQuestionPopup::questionAnswered,
                    [gamification, shimeji](const QString& answer) {
                        qDebug() << "🤔 Metacognitive response:" << answer;
                        
                        // Grant XP for thoughtful reflection
                        gamification->addXP(20, "Thoughtful metacognitive reflection");
                        
                        // Trigger grateful/proud animation for engagement
                        shimeji->triggerEmotionalAnimation(EmotionalState::Proud);
                        
                        // TODO: Send answer back to Rust consciousness system for learning
                    });
    
    QObject::connect(metacogPopup, &MetacognitiveQuestionPopup::questionDismissed,
                    []() {
                        qDebug() << "🤔 Metacognitive question dismissed";
                        // No penalty, just logging
                    });

    // Start emotion simulation (replace with actual Rust bridge signals)
    emotionTimer->start(2000); // Update every 2 seconds

    // If system tray is available, minimize main window to tray
    if (QSystemTrayIcon::isSystemTrayAvailable()) {
        QTimer::singleShot(1000, [window]() {
            if (window->isVisible()) {
                window->hide();
            }
        });
    }

    qDebug() << "🎮 Niodo Shimeji companion started successfully!";
    qDebug() << "🎯 Gamification system active - watch Niodo level up!";

    // Start event loop
    return app.exec();
}

#include "main.moc"
