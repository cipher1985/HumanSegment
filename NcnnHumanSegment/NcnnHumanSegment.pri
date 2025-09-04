#ncnn使用ModNet模型分割人物与背景
contains(DEFINES, WIN64): NcnnFolder = $$PWD/ncnn/x64
else: NcnnFolder = $$PWD/ncnn/x86

LIBS += -L$$NcnnFolder/lib/ -lncnn

INCLUDEPATH += $$NcnnFolder/include/ncnn
INCLUDEPATH += $$NcnnFolder/include
DEPENDPATH += $$NcnnFolder/include

INCLUDEPATH += $$PWD

HEADERS += \
    $$PWD/qtncnnmodnet.h \
    $$PWD/qtncnnpphumansegment.h

SOURCES += \
    $$PWD/qtncnnmodnet.cpp \
    $$PWD/qtncnnpphumansegment.cpp
