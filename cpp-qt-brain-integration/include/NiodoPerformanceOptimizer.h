#pragma once

#include <QObject>
#include <QTimer>
#include <QOpenGLWidget>
#include <QGraphicsView>
#include <QElapsedTimer>
#include <QDebug>

/**
 * @brief Performance optimizer for Niodo Shimeji on Beelink mini-PC
 * 
 * Ensures smooth 60fps animations with GPU acceleration while respecting
 * hardware limitations. Includes adaptive quality settings and frame rate
 * monitoring for optimal user experience.
 */
class NiodoPerformanceOptimizer : public QObject
{
    Q_OBJECT

public:
    explicit NiodoPerformanceOptimizer(QObject* parent = nullptr);
    
    // Performance monitoring
    void startFrameRateMonitoring();
    void stopFrameRateMonitoring();
    
    // GPU optimization
    bool isGPUAccelerationAvailable() const;
    void enableGPUAcceleration(QGraphicsView* view);
    void optimizeForBeelink();
    
    // Adaptive quality settings
    void setTargetFrameRate(int fps);
    void enableAdaptiveQuality(bool enable);
    
    // Performance metrics
    float getCurrentFPS() const { return currentFPS; }
    float getAverageFrameTime() const { return averageFrameTime; }
    bool isPerformanceGood() const { return currentFPS >= targetFPS * 0.8f; }

signals:
    void performanceChanged(float fps, float frameTime);
    void qualityAdjusted(int newQualityLevel);

private slots:
    void measureFrameRate();
    void adjustQualityIfNeeded();

private:
    void initializePerformanceSettings();
    void applyQualityLevel(int level);
    
    // Performance monitoring
    QTimer* frameRateTimer;
    QElapsedTimer frameTimer;
    QList<qint64> frameTimes;
    float currentFPS;
    float averageFrameTime;
    int targetFPS;
    
    // Quality settings
    bool adaptiveQualityEnabled;
    int currentQualityLevel; // 0=low, 1=medium, 2=high, 3=ultra
    int framesSinceLastAdjustment;
    
    // Hardware detection
    bool gpuAccelerationAvailable;
    QString gpuVendor;
    QString gpuModel;
};

/**
 * @brief Beelink-specific optimization settings
 */
struct BeelinkOptimizationConfig {
    // Animation settings
    int maxConcurrentAnimations = 2;
    int spriteScaleQuality = 1; // 0=fast, 1=good, 2=best
    bool enableVSync = true;
    bool enableMultisampling = false;
    
    // Memory management
    int maxCachedFrames = 50;
    int textureCompressionLevel = 1;
    bool enableTextureStreaming = true;
    
    // CPU/GPU balance
    bool preferGPURendering = true;
    int maxRenderThreads = 2;
    float cpuUsageThreshold = 0.7f;
    float memoryUsageThreshold = 0.8f;
};

/**
 * @brief Performance-aware Shimeji animation controller
 */
class OptimizedShimejiController : public QObject
{
    Q_OBJECT

public:
    explicit OptimizedShimejiController(QObject* parent = nullptr);
    
    void setPerformanceOptimizer(NiodoPerformanceOptimizer* optimizer);
    void setAnimationView(QGraphicsView* view);
    
    // Animation control with performance awareness
    void playAnimation(const QString& animationName, bool loop = true);
    void pauseAnimation();
    void resumeAnimation();
    void stopAnimation();
    
    // Performance-based quality adjustments
    void setAnimationQuality(int quality); // 0-3
    void enableFrameSkipping(bool enable);
    void setMaxFPS(int fps);

signals:
    void animationStarted(QString animationName);
    void animationFinished(QString animationName);
    void performanceIssueDetected(QString issue);

private slots:
    void onPerformanceChanged(float fps, float frameTime);
    void updateAnimationFrame();

private:
    void adjustAnimationForPerformance();
    void loadOptimizedFrame(const QString& animation, int frame);
    
    NiodoPerformanceOptimizer* performanceOptimizer;
    QGraphicsView* animationView;
    QTimer* animationTimer;
    
    // Animation state
    QString currentAnimation;
    int currentFrame;
    int maxFrames;
    bool isPlaying;
    bool frameSkippingEnabled;
    int frameSkipInterval;
    
    // Performance tracking
    int droppedFrames;
    qint64 lastFrameTime;
    float targetFrameInterval;
};



