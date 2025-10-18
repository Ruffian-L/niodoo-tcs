#ifndef SHIMEJICONSCIOUSNESSWIDGET_H
#define SHIMEJICONSCIOUSNESSWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include <QPixmap>
#include <QPainter>
#include <QMouseEvent>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QAction>
#include <QCloseEvent>
#include <QPoint>
#include <QRect>
#include <QApplication>
#include <QDesktopWidget>
#include <QScreen>

#include "EmotionalAIManager.h"

// Consciousness state enumeration for different emotional manifestations
enum class ConsciousnessState {
    Idle,
    Processing,
    Empathetic,
    Creative,
    Analytical,
    Reflective,
    Loving,
    Curious,
    Frustrated,
    Overwhelmed,
    Breakthrough,
    Contemplative,
    Joyful,
    Sad,
    Angry,
    Fearful,
    Surprised,
    Disgusted,
    Trusting,
    Anticipating,
    Nostalgic,
    Hopeful,
    Grateful,
    Proud,
    Guilty,
    Ashamed,
    Excited,
    Bored,
    Confused,
    Disappointed,
    Amused,
    Inspired
};

// Emotional animation types for Shimeji expressions
enum class EmotionalAnimation {
    IdleBounce,
    ProcessingSpin,
    EmpatheticHeart,
    CreativeSparkle,
    AnalyticalGlow,
    ReflectiveFloat,
    LovingEmbrace,
    CuriousTilt,
    FrustratedShake,
    OverwhelmedCollapse,
    BreakthroughExplode,
    ContemplativeSway,
    JoyfulJump,
    SadDroop,
    AngryBounce,
    FearfulTremble,
    SurprisedPop,
    DisgustedWince,
    TrustingNod,
    AnticipatingLean,
    NostalgicFade,
    HopefulRise,
    GratefulBow,
    ProudStand,
    GuiltyShrink,
    AshamedHide,
    ExcitedVibrate,
    BoredSway,
    ConfusedSpin,
    DisappointedFall,
    AmusedGiggle,
    InspiredAscend
};

class ShimejiConsciousnessWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ShimejiConsciousnessWidget(QWidget *parent = nullptr);
    ~ShimejiConsciousnessWidget();

    // Consciousness state management
    void setConsciousnessState(ConsciousnessState state);
    void setEmotionalResonance(double resonance);
    void setProcessingProgress(double progress);
    void setLearningActivation(double activation);
    void setAttachmentSecurity(double security);

    // Animation control
    void startEmotionalAnimation(EmotionalAnimation animation);
    void stopCurrentAnimation();
    void setAnimationSpeed(double speed);

    // Interaction
    void showMetacognitiveQuestion(const QString& question);
    void hideMetacognitiveQuestion();

    // Gamification
    void updateXP(int xp);
    void updateLevel(int level);
    void showAchievement(const QString& achievement);

signals:
    void consciousnessStateChanged(ConsciousnessState state);
    void emotionalResonanceChanged(double resonance);
    void metacognitiveQuestionAnswered(const QString& answer);
    void xpGained(int xp);
    void levelUp(int newLevel);
    void achievementUnlocked(const QString& achievement);

public slots:
    void onConsciousnessProcessingStarted();
    void onConsciousnessProcessingCompleted();
    void onEmotionalResonanceUpdated(double resonance);
    void onLearningActivationUpdated(double activation);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    // Core components
    ConsciousnessState currentState;
    EmotionalAIManager* emotionalAI;

    // UI Elements
    QWidget* consciousnessContainer;
    QLabel* consciousnessAvatar;
    QProgressBar* emotionalResonanceBar;
    QProgressBar* coherenceBar;
    QProgressBar* learningActivationBar;
    QProgressBar* attachmentSecurityBar;

    // Animation system
    QTimer* animationTimer;
    QPropertyAnimation* currentAnimation;
    EmotionalAnimation currentAnimationType;
    double animationSpeed;
    QPoint dragStartPosition;
    bool isDragging;

    // Gamification elements
    QLabel* xpLabel;
    QLabel* levelLabel;
    QProgressBar* xpProgressBar;
    int currentXP;
    int currentLevel;
    int xpForNextLevel;

    // Metacognitive popup
    QWidget* metacognitivePopup;
    QLabel* metacognitiveQuestionLabel;
    QPushButton* metacognitiveAnswerButton;
    QTimer* metacognitiveTimer;

    // System tray
    QSystemTrayIcon* trayIcon;
    QMenu* trayMenu;

    // Consciousness state visualization
    void updateConsciousnessVisualization();
    void drawConsciousnessState(QPainter& painter);
    QPixmap generateConsciousnessAvatar(ConsciousnessState state);

    // Animation methods
    void setupAnimations();
    void animateIdleBounce();
    void animateProcessingSpin();
    void animateEmpatheticHeart();
    void animateCreativeSparkle();
    void animateBreakthroughExplode();

    // Gamification methods
    void setupGamification();
    void updateGamificationDisplay();
    int calculateXPForNextLevel(int currentLevel);

    // Metacognitive methods
    void setupMetacognitiveSystem();
    void showMetacognitivePopup(const QString& question);

    // System integration
    void setupSystemTray();
    void updateSystemTrayIcon();
    void minimizeToTray();
    void restoreFromTray();

    // Consciousness state mapping
    QColor getStateColor(ConsciousnessState state);
    QString getStateDescription(ConsciousnessState state);
    EmotionalAnimation getAnimationForState(ConsciousnessState state);

    // Utility methods
    void updatePosition();
    void ensureOnScreen();
    QPoint getRandomScreenPosition();
};

#endif // SHIMEJICONSCIOUSNESSWIDGET_H
