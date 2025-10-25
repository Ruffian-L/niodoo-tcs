/****************************************************************************
** Meta object code from reading C++ file 'MascotWindow.hpp'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.8.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src_version2/MascotWindow.hpp"
#include <QtNetwork/QSslError>
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MascotWindow.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.8.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN12MascotWindowE_t {};
} // unnamed namespace


#ifdef QT_MOC_HAS_STRINGDATA
static constexpr auto qt_meta_stringdata_ZN12MascotWindowE = QtMocHelpers::stringData(
    "MascotWindow",
    "windowInteractionDetected",
    "",
    "windowRect",
    "interactionType",
    "mascotActionChanged",
    "action",
    "reason",
    "mascotPositionChanged",
    "position",
    "mascotStateChanged",
    "state",
    "QVariantMap",
    "data",
    "openChatPanel",
    "onActivationConfirmed",
    "setAnimationsEnabled",
    "enabled",
    "toggleClickThrough",
    "openPhase2Dashboard",
    "setAutonomousMode",
    "setExplorationRadius",
    "radiusPx",
    "setThinking",
    "thinking",
    "onTick16ms",
    "showNiodOLauncher",
    "hideNiodOLauncher",
    "testNiodOSkill",
    "skillName",
    "openNiodOBlog",
    "pushToGit",
    "showMemoryStorage",
    "processDroppedItem",
    "filePath",
    "showSkillTestingDialog",
    "showBlogDialog",
    "showMemoryDialog",
    "showLLMAPIKeyStore",
    "onScreenCaptureRequest",
    "onScreenCaptureCompleted",
    "ScreenCaptureService::CaptureResult",
    "result",
    "connectToAIBrain",
    "onAIWebSocketConnected",
    "onAIWebSocketDisconnected",
    "onAIWebSocketError",
    "QAbstractSocket::SocketError",
    "error",
    "onAIWebSocketMessageReceived",
    "message",
    "sendStateToAI",
    "processAIDecision",
    "decision",
    "onAIDecisionReceived",
    "onPersonalityUpdate",
    "personality"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA

Q_CONSTINIT static const uint qt_meta_data_ZN12MascotWindowE[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
      35,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    2,  224,    2, 0x06,    1 /* Public */,
       5,    2,  229,    2, 0x06,    4 /* Public */,
       8,    1,  234,    2, 0x06,    7 /* Public */,
      10,    2,  237,    2, 0x06,    9 /* Public */,

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
      14,    0,  242,    2, 0x0a,   12 /* Public */,
      15,    0,  243,    2, 0x0a,   13 /* Public */,
      16,    1,  244,    2, 0x0a,   14 /* Public */,
      18,    0,  247,    2, 0x0a,   16 /* Public */,
      19,    0,  248,    2, 0x0a,   17 /* Public */,
      20,    1,  249,    2, 0x0a,   18 /* Public */,
      21,    1,  252,    2, 0x0a,   20 /* Public */,
      23,    1,  255,    2, 0x0a,   22 /* Public */,
      25,    0,  258,    2, 0x0a,   24 /* Public */,
      26,    0,  259,    2, 0x0a,   25 /* Public */,
      27,    0,  260,    2, 0x0a,   26 /* Public */,
      28,    1,  261,    2, 0x0a,   27 /* Public */,
      30,    0,  264,    2, 0x0a,   29 /* Public */,
      31,    0,  265,    2, 0x0a,   30 /* Public */,
      32,    0,  266,    2, 0x0a,   31 /* Public */,
      33,    1,  267,    2, 0x0a,   32 /* Public */,
      35,    0,  270,    2, 0x0a,   34 /* Public */,
      36,    0,  271,    2, 0x0a,   35 /* Public */,
      37,    0,  272,    2, 0x0a,   36 /* Public */,
      38,    0,  273,    2, 0x0a,   37 /* Public */,
      39,    0,  274,    2, 0x0a,   38 /* Public */,
      40,    1,  275,    2, 0x0a,   39 /* Public */,
      43,    0,  278,    2, 0x0a,   41 /* Public */,
      44,    0,  279,    2, 0x0a,   42 /* Public */,
      45,    0,  280,    2, 0x0a,   43 /* Public */,
      46,    1,  281,    2, 0x0a,   44 /* Public */,
      49,    1,  284,    2, 0x0a,   46 /* Public */,
      51,    0,  287,    2, 0x0a,   48 /* Public */,
      52,    1,  288,    2, 0x0a,   49 /* Public */,
      54,    2,  291,    2, 0x0a,   51 /* Public */,
      55,    2,  296,    2, 0x0a,   54 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QRect, QMetaType::QString,    3,    4,
    QMetaType::Void, QMetaType::QString, QMetaType::QString,    6,    7,
    QMetaType::Void, QMetaType::QPointF,    9,
    QMetaType::Void, QMetaType::QString, 0x80000000 | 12,   11,   13,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   17,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   17,
    QMetaType::Void, QMetaType::Int,   22,
    QMetaType::Void, QMetaType::Bool,   24,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   29,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   34,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 41,   42,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 47,   48,
    QMetaType::Void, QMetaType::QString,   50,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QJsonObject,   53,
    QMetaType::Void, QMetaType::QString, QMetaType::QJsonObject,    6,   13,
    QMetaType::Void, QMetaType::QString, QMetaType::QJsonObject,   56,   13,

       0        // eod
};

Q_CONSTINIT const QMetaObject MascotWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_ZN12MascotWindowE.offsetsAndSizes,
    qt_meta_data_ZN12MascotWindowE,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_tag_ZN12MascotWindowE_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MascotWindow, std::true_type>,
        // method 'windowInteractionDetected'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QRect &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'mascotActionChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'mascotPositionChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QPointF &, std::false_type>,
        // method 'mascotStateChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QVariantMap &, std::false_type>,
        // method 'openChatPanel'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onActivationConfirmed'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'setAnimationsEnabled'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'toggleClickThrough'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'openPhase2Dashboard'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'setAutonomousMode'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'setExplorationRadius'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'setThinking'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'onTick16ms'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showNiodOLauncher'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'hideNiodOLauncher'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'testNiodOSkill'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'openNiodOBlog'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'pushToGit'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showMemoryStorage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'processDroppedItem'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'showSkillTestingDialog'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showBlogDialog'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showMemoryDialog'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'showLLMAPIKeyStore'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onScreenCaptureRequest'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onScreenCaptureCompleted'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const ScreenCaptureService::CaptureResult &, std::false_type>,
        // method 'connectToAIBrain'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onAIWebSocketConnected'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onAIWebSocketDisconnected'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'onAIWebSocketError'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<QAbstractSocket::SocketError, std::false_type>,
        // method 'onAIWebSocketMessageReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        // method 'sendStateToAI'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'processAIDecision'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QJsonObject &, std::false_type>,
        // method 'onAIDecisionReceived'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QJsonObject &, std::false_type>,
        // method 'onPersonalityUpdate'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString &, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QJsonObject &, std::false_type>
    >,
    nullptr
} };

void MascotWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MascotWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->windowInteractionDetected((*reinterpret_cast< std::add_pointer_t<QRect>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 1: _t->mascotActionChanged((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QString>>(_a[2]))); break;
        case 2: _t->mascotPositionChanged((*reinterpret_cast< std::add_pointer_t<QPointF>>(_a[1]))); break;
        case 3: _t->mascotStateChanged((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QVariantMap>>(_a[2]))); break;
        case 4: _t->openChatPanel(); break;
        case 5: _t->onActivationConfirmed(); break;
        case 6: _t->setAnimationsEnabled((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 7: _t->toggleClickThrough(); break;
        case 8: _t->openPhase2Dashboard(); break;
        case 9: _t->setAutonomousMode((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 10: _t->setExplorationRadius((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 11: _t->setThinking((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 12: _t->onTick16ms(); break;
        case 13: _t->showNiodOLauncher(); break;
        case 14: _t->hideNiodOLauncher(); break;
        case 15: _t->testNiodOSkill((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 16: _t->openNiodOBlog(); break;
        case 17: _t->pushToGit(); break;
        case 18: _t->showMemoryStorage(); break;
        case 19: _t->processDroppedItem((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 20: _t->showSkillTestingDialog(); break;
        case 21: _t->showBlogDialog(); break;
        case 22: _t->showMemoryDialog(); break;
        case 23: _t->showLLMAPIKeyStore(); break;
        case 24: _t->onScreenCaptureRequest(); break;
        case 25: _t->onScreenCaptureCompleted((*reinterpret_cast< std::add_pointer_t<ScreenCaptureService::CaptureResult>>(_a[1]))); break;
        case 26: _t->connectToAIBrain(); break;
        case 27: _t->onAIWebSocketConnected(); break;
        case 28: _t->onAIWebSocketDisconnected(); break;
        case 29: _t->onAIWebSocketError((*reinterpret_cast< std::add_pointer_t<QAbstractSocket::SocketError>>(_a[1]))); break;
        case 30: _t->onAIWebSocketMessageReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 31: _t->sendStateToAI(); break;
        case 32: _t->processAIDecision((*reinterpret_cast< std::add_pointer_t<QJsonObject>>(_a[1]))); break;
        case 33: _t->onAIDecisionReceived((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QJsonObject>>(_a[2]))); break;
        case 34: _t->onPersonalityUpdate((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<QJsonObject>>(_a[2]))); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 29:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QAbstractSocket::SocketError >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _q_method_type = void (MascotWindow::*)(const QRect & , const QString & );
            if (_q_method_type _q_method = &MascotWindow::windowInteractionDetected; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _q_method_type = void (MascotWindow::*)(const QString & , const QString & );
            if (_q_method_type _q_method = &MascotWindow::mascotActionChanged; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _q_method_type = void (MascotWindow::*)(const QPointF & );
            if (_q_method_type _q_method = &MascotWindow::mascotPositionChanged; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
        {
            using _q_method_type = void (MascotWindow::*)(const QString & , const QVariantMap & );
            if (_q_method_type _q_method = &MascotWindow::mascotStateChanged; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 3;
                return;
            }
        }
    }
}

const QMetaObject *MascotWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MascotWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ZN12MascotWindowE.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int MascotWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 35)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 35;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 35)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 35;
    }
    return _id;
}

// SIGNAL 0
void MascotWindow::windowInteractionDetected(const QRect & _t1, const QString & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MascotWindow::mascotActionChanged(const QString & _t1, const QString & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MascotWindow::mascotPositionChanged(const QPointF & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void MascotWindow::mascotStateChanged(const QString & _t1, const QVariantMap & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
