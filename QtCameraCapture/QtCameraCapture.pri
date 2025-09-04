QT       += multimediawidgets

greaterThan(QT_MAJOR_VERSION, 5) : QtCameraCapture = $$PWD/qt6
else : QtCameraCapture = $$PWD/qt5

HEADERS += \
    $$QtCameraCapture/qtcameracapture.h

SOURCES += \
    $$QtCameraCapture/qtcameracapture.cpp

INCLUDEPATH += $$QtCameraCapture
