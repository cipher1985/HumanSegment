include(onnxruntime/onnxruntime.pri)

HEADERS += \
    $$PWD/qtonnxmodnet.h \
    $$PWD/qtonnxpphumansegment.h \
    $$PWD/qtonnxrobustvideomatting.h

SOURCES += \
    $$PWD/qtonnxmodnet.cpp \
    $$PWD/qtonnxpphumansegment.cpp \
    $$PWD/qtonnxrobustvideomatting.cpp

INCLUDEPATH += $$PWD
