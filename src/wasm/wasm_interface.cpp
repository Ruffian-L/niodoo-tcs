/*
 * 🌐 WASM Interface - Browser-Native Qt 3D Visualization
 *
 * WebAssembly interface for Qt 3D visualization with advanced memory management
 * for MoE-like emotional perspectives in NiodO.o consciousness engine.
 */

#include "wasm_interface.h"
#include "emotional_visualization.h"
#include "mobius_memory_viz.h"
#include <emscripten.h>
#include <emscripten/bind.h>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQuickView>
#include <QWebEngineView>
#include <QTimer>
#include <memory>

// Global application instance for WASM
static std::unique_ptr<QApplication> wasmApp;
static std::unique_ptr<QQmlApplicationEngine> wasmEngine;
static std::unique_ptr<EmotionalVisualization> emotionalViz;
static std::unique_ptr<MobiusMemoryVisualization> mobiusViz;

// Memory management for WASM
static MemoryManager* memoryManager = nullptr;

// WASM exports for JavaScript interop
extern "C" {

    EMSCRIPTEN_KEEPALIVE
    int initializeVisualization() {
        try {
            // Initialize Qt Application for WASM
            if (!wasmApp) {
                int argc = 0;
                wasmApp = std::make_unique<QApplication>(argc, nullptr);
            }

            // Initialize QML engine
            if (!wasmEngine) {
                wasmEngine = std::make_unique<QQmlApplicationEngine>();

                // Register QML types
                qmlRegisterType<EmotionalVisualization>("NiodOo.Emotional", 1, 0, "EmotionalVisualization");
                qmlRegisterType<MobiusMemoryVisualization>("NiodOo.Memory", 1, 0, "MobiusMemoryVisualization");

                // Load main QML file
                wasmEngine->load(QUrl("qrc:/qml/EmotionalVisualization.qml"));
            }

            // Initialize emotional visualization
            if (!emotionalViz) {
                emotionalViz = std::make_unique<EmotionalVisualization>();
            }

            // Initialize Möbius memory visualization
            if (!mobiusViz) {
                mobiusViz = std::make_unique<MobiusMemoryVisualization>();
            }

            // Initialize memory manager
            if (!memoryManager) {
                memoryManager = new MemoryManager();
            }

            return 0; // Success
        } catch (const std::exception& e) {
            printf("Error initializing visualization: %s\n", e.what());
            return -1; // Error
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int updateEmotionalState(const char* emotionalDataJson) {
        if (!emotionalViz) {
            return -1;
        }

        try {
            QString jsonStr(emotionalDataJson);
            QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());

            if (doc.isNull()) {
                return -2; // Invalid JSON
            }

            QJsonObject obj = doc.object();

            EmotionalState state;
            state.valence = obj.value("valence").toDouble();
            state.arousal = obj.value("arousal").toDouble();
            state.dominance = obj.value("dominance").toDouble();
            state.emotionalIntensity = obj.value("intensity").toDouble();

            emotionalViz->updateEmotionalState(state);
            return 0;
        } catch (const std::exception& e) {
            printf("Error updating emotional state: %s\n", e.what());
            return -1;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void renderFrame() {
        if (emotionalViz) {
            emotionalViz->renderFrame();
        }

        if (mobiusViz) {
            mobiusViz->renderFrame();
        }

        // Manage memory for inactive "experts" (MoE-like memory management)
        if (memoryManager) {
            memoryManager->manageMemory();
        }
    }

    EMSCRIPTEN_KEEPALIVE
    const char* getVisualizationMetrics() {
        static QString metricsJson;

        if (!emotionalViz && !mobiusViz) {
            return "{}";
        }

        QJsonObject metrics;

        if (emotionalViz) {
            QJsonObject emotionalMetrics;
            emotionalMetrics["fps"] = emotionalViz->getFPS();
            emotionalMetrics["memoryUsage"] = emotionalViz->getMemoryUsage();
            emotionalMetrics["emotionalCoherence"] = emotionalViz->getEmotionalCoherence();
            metrics["emotional"] = emotionalMetrics;
        }

        if (mobiusViz) {
            QJsonObject mobiusMetrics;
            mobiusMetrics["topologyPreservation"] = mobiusViz->getTopologyPreservation();
            mobiusMetrics["memoryEfficiency"] = mobiusViz->getMemoryEfficiency();
            mobiusMetrics["activeExperts"] = mobiusViz->getActiveExpertCount();
            metrics["mobius"] = mobiusMetrics;
        }

        if (memoryManager) {
            QJsonObject memoryMetrics;
            memoryMetrics["totalMemory"] = memoryManager->getTotalMemoryUsage();
            memoryMetrics["expertMemory"] = memoryManager->getExpertMemoryUsage();
            memoryMetrics["inactiveExperts"] = memoryManager->getInactiveExpertCount();
            metrics["memory"] = memoryMetrics;
        }

        metricsJson = QString(QJsonDocument(metrics).toJson(QJsonDocument::Compact));
        return metricsJson.toUtf8().constData();
    }

    EMSCRIPTEN_KEEPALIVE
    void setMemoryManagementMode(int mode) {
        if (memoryManager) {
            memoryManager->setMode(static_cast<MemoryManagementMode>(mode));
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void triggerMemoryOptimization() {
        if (memoryManager) {
            memoryManager->optimizeMemory();
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int getMemoryPressure() {
        if (memoryManager) {
            return memoryManager->getMemoryPressure();
        }
        return 0;
    }
}

// Emscripten bindings for JavaScript
EMSCRIPTEN_BINDINGS(niodoo_wasm) {
    emscripten::function("initializeVisualization", &initializeVisualization);
    emscripten::function("updateEmotionalState", &updateEmotionalState);
    emscripten::function("renderFrame", &renderFrame);
    emscripten::function("getVisualizationMetrics", &getVisualizationMetrics);
    emscripten::function("setMemoryManagementMode", &setMemoryManagementMode);
    emscripten::function("triggerMemoryOptimization", &triggerMemoryOptimization);
    emscripten::function("getMemoryPressure", &getMemoryPressure);

    // Register string return type for getVisualizationMetrics
    emscripten::register_vector<std::string>("vector<string>");
}

// Advanced memory management for WASM
MemoryManager::MemoryManager() : mode(MemoryManagementMode::Balanced), totalMemoryUsage(0) {
    // Initialize memory tracking
    updateMemoryUsage();
}

void MemoryManager::manageMemory() {
    updateMemoryUsage();

    switch (mode) {
        case MemoryManagementMode::Conservative:
            manageConservative();
            break;
        case MemoryManagementMode::Balanced:
            manageBalanced();
            break;
        case MemoryManagementMode::Aggressive:
            manageAggressive();
            break;
    }
}

void MemoryManager::updateMemoryUsage() {
    // Get memory usage from browser (emscripten_get_heap_size)
    totalMemoryUsage = emscripten_get_heap_size();

    // Update expert memory tracking
    updateExpertMemory();
}

void MemoryManager::manageConservative() {
    // Conservative mode: only unload clearly inactive experts
    const size_t threshold = 50 * 1024 * 1024; // 50MB

    if (totalMemoryUsage > threshold) {
        // Find and unload least recently used experts
        unloadInactiveExperts(0.3f); // Unload 30% of inactive experts
    }
}

void MemoryManager::manageBalanced() {
    // Balanced mode: moderate memory management
    const size_t threshold = 75 * 1024 * 1024; // 75MB

    if (totalMemoryUsage > threshold) {
        // Balance between memory usage and performance
        unloadInactiveExperts(0.5f); // Unload 50% of inactive experts
    }
}

void MemoryManager::manageAggressive() {
    // Aggressive mode: prioritize memory over performance
    const size_t threshold = 100 * 1024 * 1024; // 100MB

    if (totalMemoryUsage > threshold) {
        // Aggressively free memory
        unloadInactiveExperts(0.8f); // Unload 80% of inactive experts
        forceGarbageCollection();
    }
}

void MemoryManager::unloadInactiveExperts(float ratio) {
    // Sort experts by last access time
    std::sort(expertMemory.begin(), expertMemory.end(),
              [](const ExpertMemoryInfo& a, const ExpertMemoryInfo& b) {
                  return a.lastAccessTime < b.lastAccessTime;
              });

    // Unload experts based on ratio
    size_t expertsToUnload = static_cast<size_t>(expertMemory.size() * ratio);

    for (size_t i = 0; i < expertsToUnload && i < expertMemory.size(); ++i) {
        ExpertMemoryInfo& expert = expertMemory[i];

        // Free expert memory
        if (expert.data) {
            free(expert.data);
            expert.data = nullptr;
            expert.memorySize = 0;
        }

        printf("Unloaded inactive expert: %s\n", expert.expertId.c_str());
    }

    // Remove unloaded experts from tracking
    expertMemory.erase(expertMemory.begin(),
                      expertMemory.begin() + expertsToUnload);
}

void MemoryManager::updateExpertMemory() {
    // Update memory usage for tracked experts
    size_t totalExpertMemory = 0;

    for (const auto& expert : expertMemory) {
        totalExpertMemory += expert.memorySize;
    }

    expertMemoryUsage = totalExpertMemory;
}

void MemoryManager::forceGarbageCollection() {
    // Force JavaScript garbage collection through Emscripten
    EM_ASM({
        if (typeof gc !== 'undefined') {
            gc();
        }
    });
}

void MemoryManager::optimizeMemory() {
    // Perform comprehensive memory optimization
    printf("Performing memory optimization...\n");

    // Force garbage collection
    forceGarbageCollection();

    // Compact memory if possible
    // (In a real implementation, this would reorganize memory layout)

    // Update memory statistics after optimization
    updateMemoryUsage();

    printf("Memory optimization completed. Total usage: %zu bytes\n", totalMemoryUsage);
}

// WASM-specific Qt setup
void setupWasmQt() {
    // Configure Qt for WebAssembly environment
    qputenv("QT_QPA_PLATFORM", "wasm");
    qputenv("QT_WEBGL_VERSION", "webgl2");

    // Enable WebGL for 3D rendering
    qputenv("QT_OPENGL", "angle");

    // Configure memory management for WASM
    qputenv("QT_WASM_MEMORY", "256"); // 256MB heap
}
