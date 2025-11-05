#include "NiodoPerformanceOptimizer.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QApplication>
#include <QScreen>
#include <QSysInfo>
#include <QProcess>
#include <QRegularExpression>

NiodoPerformanceOptimizer::NiodoPerformanceOptimizer(QObject* parent)
    : QObject(parent)
    , frameRateTimer(new QTimer(this))
    , currentFPS(60.0f)
    , averageFrameTime(16.67f)
    , targetFPS(60)
    , adaptiveQualityEnabled(true)
    , currentQualityLevel(2) // Start with high quality
    , framesSinceLastAdjustment(0)
    , gpuAccelerationAvailable(false)
{
    initializePerformanceSettings();
    
    connect(frameRateTimer, &QTimer::timeout, this, &NiodoPerformanceOptimizer::measureFrameRate);
    connect(frameRateTimer, &QTimer::timeout, this, &NiodoPerformanceOptimizer::adjustQualityIfNeeded);
}

void NiodoPerformanceOptimizer::initializePerformanceSettings()
{
    qDebug() << "🎮 Initializing Niodo Performance Optimizer for Beelink";
    
    // Detect GPU capabilities
    gpuAccelerationAvailable = QOpenGLContext::openGLModuleType() != QOpenGLContext::LibGL;
    
    if (gpuAccelerationAvailable) {
        QOpenGLContext context;
        if (context.create()) {
            QOpenGLFunctions* gl = context.functions();
            if (gl) {
                gpuVendor = QString((const char*)gl->glGetString(GL_VENDOR));
                gpuModel = QString((const char*)gl->glGetString(GL_RENDERER));
                qDebug() << "🎯 GPU detected:" << gpuVendor << gpuModel;
            }
        }
    }
    
    // Optimize for Beelink hardware profile
    optimizeForBeelink();
    
    frameTimer.start();
}

void NiodoPerformanceOptimizer::optimizeForBeelink()
{
    qDebug() << "🔧 Applying Beelink mini-PC optimizations";
    
    BeelinkOptimizationConfig config;
    
    // Detect if we're likely on a Beelink (or similar mini-PC)
    QString systemInfo = QSysInfo::prettyProductName();
    bool isMiniPC = systemInfo.contains("Mini", Qt::CaseInsensitive) || 
                    systemInfo.contains("NUC", Qt::CaseInsensitive) ||
                    systemInfo.contains("Beelink", Qt::CaseInsensitive);
    
    if (isMiniPC) {
        qDebug() << "📱 Mini-PC detected, applying conservative settings";
        
        // Conservative settings for mini-PC
        targetFPS = 30; // Reduce target to 30fps for stability
        currentQualityLevel = 1; // Start with medium quality
        config.maxConcurrentAnimations = 1;
        config.enableMultisampling = false;
        config.maxRenderThreads = 1;
    } else {
        qDebug() << "🖥️ Desktop/laptop detected, using performance settings";
        
        // More aggressive settings for full PCs
        targetFPS = 60;
        currentQualityLevel = 2;
    }
    
    // Apply the configuration
    applyQualityLevel(currentQualityLevel);
}

bool NiodoPerformanceOptimizer::isGPUAccelerationAvailable() const
{
    return gpuAccelerationAvailable;
}

void NiodoPerformanceOptimizer::enableGPUAcceleration(QGraphicsView* view)
{
    if (!isGPUAccelerationAvailable() || !view) {
        qWarning() << "❌ GPU acceleration not available or invalid view";
        return;
    }
    
    // Create OpenGL viewport for hardware acceleration
    QOpenGLWidget* glWidget = new QOpenGLWidget();
    
    // Configure OpenGL for optimal Shimeji rendering
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    
    // Enable multisampling only on higher-end hardware
    if (currentQualityLevel >= 2 && !gpuModel.contains("Intel", Qt::CaseInsensitive)) {
        format.setSamples(4);
    }
    
    glWidget->setFormat(format);
    view->setViewport(glWidget);
    
    // Optimize rendering hints for Shimeji sprites
    view->setRenderHint(QPainter::Antialiasing, currentQualityLevel >= 1);
    view->setRenderHint(QPainter::SmoothPixmapTransform, currentQualityLevel >= 2);
    view->setRenderHint(QPainter::HighQualityAntialiasing, currentQualityLevel >= 3);
    
    // Enable caching for better performance
    view->setCacheMode(QGraphicsView::CacheBackground);
    view->setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);
    
    qDebug() << "✅ GPU acceleration enabled with quality level" << currentQualityLevel;
}

void NiodoPerformanceOptimizer::startFrameRateMonitoring()
{
    frameTimes.clear();
    frameRateTimer->start(1000); // Check every second
    frameTimer.restart();
    
    qDebug() << "📊 Frame rate monitoring started (target:" << targetFPS << "fps)";
}

void NiodoPerformanceOptimizer::stopFrameRateMonitoring()
{
    frameRateTimer->stop();
    qDebug() << "📊 Frame rate monitoring stopped";
}

void NiodoPerformanceOptimizer::measureFrameRate()
{
    qint64 elapsed = frameTimer.restart();
    frameTimes.append(elapsed);
    
    // Keep only last 60 frame times (1 second at 60fps)
    if (frameTimes.size() > 60) {
        frameTimes.removeFirst();
    }
    
    // Calculate current FPS
    if (!frameTimes.isEmpty()) {
        qint64 totalTime = 0;
        for (qint64 time : frameTimes) {
            totalTime += time;
        }
        
        averageFrameTime = totalTime / (float)frameTimes.size();
        currentFPS = 1000.0f / averageFrameTime;
        
        emit performanceChanged(currentFPS, averageFrameTime);
    }
    
    framesSinceLastAdjustment++;
}

void NiodoPerformanceOptimizer::adjustQualityIfNeeded()
{
    if (!adaptiveQualityEnabled || framesSinceLastAdjustment < 5) {
        return; // Wait at least 5 seconds between adjustments
    }
    
    bool performanceGood = isPerformanceGood();
    int newQualityLevel = currentQualityLevel;
    
    if (!performanceGood && currentQualityLevel > 0) {
        // Performance is poor, reduce quality
        newQualityLevel = currentQualityLevel - 1;
        qDebug() << "⬇️ Reducing quality to level" << newQualityLevel << "(FPS:" << currentFPS << ")";
    } else if (performanceGood && currentQualityLevel < 3 && currentFPS > targetFPS * 1.2f) {
        // Performance is excellent, try increasing quality
        newQualityLevel = currentQualityLevel + 1;
        qDebug() << "⬆️ Increasing quality to level" << newQualityLevel << "(FPS:" << currentFPS << ")";
    }
    
    if (newQualityLevel != currentQualityLevel) {
        applyQualityLevel(newQualityLevel);
        currentQualityLevel = newQualityLevel;
        framesSinceLastAdjustment = 0;
        emit qualityAdjusted(newQualityLevel);
    }
}

void NiodoPerformanceOptimizer::applyQualityLevel(int level)
{
    qDebug() << "🎨 Applying quality level:" << level;
    
    // Quality levels:
    // 0 = Low: Fast rendering, minimal effects
    // 1 = Medium: Balanced quality/performance
    // 2 = High: Good quality with some effects
    // 3 = Ultra: Maximum quality (may impact performance)
    
    // These settings would be applied to the actual rendering system
    // For now, we just store the level for other components to use
}

void NiodoPerformanceOptimizer::setTargetFrameRate(int fps)
{
    targetFPS = qBound(15, fps, 120);
    qDebug() << "🎯 Target frame rate set to:" << targetFPS << "fps";
}

void NiodoPerformanceOptimizer::enableAdaptiveQuality(bool enable)
{
    adaptiveQualityEnabled = enable;
    qDebug() << "🔄 Adaptive quality:" << (enable ? "enabled" : "disabled");
}

// OptimizedShimejiController implementation

OptimizedShimejiController::OptimizedShimejiController(QObject* parent)
    : QObject(parent)
    , performanceOptimizer(nullptr)
    , animationView(nullptr)
    , animationTimer(new QTimer(this))
    , currentFrame(0)
    , maxFrames(0)
    , isPlaying(false)
    , frameSkippingEnabled(false)
    , frameSkipInterval(1)
    , droppedFrames(0)
    , lastFrameTime(0)
    , targetFrameInterval(83.33f) // ~12fps for sprite animations
{
    connect(animationTimer, &QTimer::timeout, this, &OptimizedShimejiController::updateAnimationFrame);
    
    // Set conservative animation timing for Beelink
    animationTimer->setInterval(83); // ~12fps for smooth sprite animation
}

void OptimizedShimejiController::setPerformanceOptimizer(NiodoPerformanceOptimizer* optimizer)
{
    performanceOptimizer = optimizer;
    if (optimizer) {
        connect(optimizer, &NiodoPerformanceOptimizer::performanceChanged,
                this, &OptimizedShimejiController::onPerformanceChanged);
    }
}

void OptimizedShimejiController::setAnimationView(QGraphicsView* view)
{
    animationView = view;
}

void OptimizedShimejiController::playAnimation(const QString& animationName, bool loop)
{
    currentAnimation = animationName;
    currentFrame = 0;
    isPlaying = true;
    
    // Adjust timing based on performance
    if (performanceOptimizer && !performanceOptimizer->isPerformanceGood()) {
        // Reduce frame rate if performance is poor
        animationTimer->setInterval(100); // 10fps
        frameSkippingEnabled = true;
        frameSkipInterval = 2;
    } else {
        // Normal frame rate
        animationTimer->setInterval(83); // 12fps
        frameSkippingEnabled = false;
        frameSkipInterval = 1;
    }
    
    animationTimer->start();
    emit animationStarted(animationName);
    
    qDebug() << "🎭 Started animation:" << animationName 
             << "(fps:" << (frameSkippingEnabled ? "reduced" : "normal") << ")";
}

void OptimizedShimejiController::onPerformanceChanged(float fps, float frameTime)
{
    // Adjust animation quality based on performance
    if (fps < 20.0f && !frameSkippingEnabled) {
        frameSkippingEnabled = true;
        frameSkipInterval = 2;
        emit performanceIssueDetected("Low FPS detected, enabling frame skipping");
    } else if (fps > 45.0f && frameSkippingEnabled) {
        frameSkippingEnabled = false;
        frameSkipInterval = 1;
    }
}

void OptimizedShimejiController::updateAnimationFrame()
{
    if (!isPlaying || currentAnimation.isEmpty()) {
        return;
    }
    
    // Frame skipping logic
    static int skipCounter = 0;
    if (frameSkippingEnabled) {
        skipCounter++;
        if (skipCounter < frameSkipInterval) {
            return;
        }
        skipCounter = 0;
    }
    
    // Load and display the frame
    loadOptimizedFrame(currentAnimation, currentFrame);
    
    // Advance to next frame
    currentFrame++;
    if (currentFrame >= maxFrames) {
        currentFrame = 0;
        emit animationFinished(currentAnimation);
    }
    
    lastFrameTime = QDateTime::currentMSecsSinceEpoch();
}

void OptimizedShimejiController::loadOptimizedFrame(const QString& animation, int frame)
{
    // This would load the actual sprite frame with performance optimizations
    // For now, this is a placeholder
    Q_UNUSED(animation)
    Q_UNUSED(frame)
}

void OptimizedShimejiController::setMaxFPS(int fps)
{
    int interval = 1000 / qBound(5, fps, 60);
    animationTimer->setInterval(interval);
    targetFrameInterval = interval;
    
    qDebug() << "🎬 Animation FPS set to:" << fps << "(interval:" << interval << "ms)";
}



