// main_qt.cpp - Qt application entry point for Gaussian visualization
#include <QtCore/QCoreApplication>
#include <QtGui/QGuiApplication>
#include <QtQml/QQmlApplicationEngine>
#include <QtCore/QUrl>
#include <QtCore/QObject>
#include <QtPlugin>

// Import the static QML plugin
Q_IMPORT_PLUGIN(com_niodoo_gaussian_plugin)

int main(int argc, char* argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;

    const QUrl url(QStringLiteral("file:///home/ruffian/Desktop/Projects/Niodoo-Feeling/qml/GaussianMemoryViz.qml"));

    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject* obj, const QUrl& objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}