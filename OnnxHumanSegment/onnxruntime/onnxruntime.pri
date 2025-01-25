contains(DEFINES, WIN64): OnnxFolder = $$PWD/x64
else: OnnxFolder = $$PWD/x86

LIBS += -L$$OnnxFolder/lib/ -lonnxruntime

INCLUDEPATH += $$OnnxFolder/include
DEPENDPATH += $$OnnxFolder/include

INCLUDEPATH += $$PWD

