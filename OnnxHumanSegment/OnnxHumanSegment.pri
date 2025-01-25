include(onnxruntime/onnxruntime.pri)

HEADERS += \
    $$PWD/qtonnxmodnet.h \
    $$PWD/qtonnxrobustvideomatting.h

SOURCES += \
    $$PWD/qtonnxmodnet.cpp \
    $$PWD/qtonnxrobustvideomatting.cpp

INCLUDEPATH += $$PWD
