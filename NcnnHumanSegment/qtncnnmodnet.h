#ifndef QTNCNNMODNET_H
#define QTNCNNMODNET_H

#include <QObject>
#include <QMutex>
#include <QImage>

namespace ncnn {
    class Net;
}
//使用ModNet模型分割人物与背景
class QtNcnnModNet : public QObject
{
    Q_OBJECT
public:
    explicit QtNcnnModNet(QObject *parent,
        const QString& paramFile = "modnet_webcam_portrait_matting_256x256.param",
        const QString& modelFile = "modnet_webcam_portrait_matting_256x256.bin",
        const QString& inputName = "input",
        const QString& outputName = "output",
        int modelW = 256, int modelH = 256);
    explicit QtNcnnModNet(
        const QString& paramFile = "modnet_webcam_portrait_matting_256x256.param",
        const QString& modelFile = "modnet_webcam_portrait_matting_256x256.bin",
        const QString& inputName = "input",
        const QString& outputName = "output",
        int modelW = 256, int modelH = 256, QObject *parent = nullptr);
    // 获得分割图像
    QImage segmentImage(QImage imgBgra, QImage* retMask = nullptr);
private:
    //初始化ncnn组件
    bool init();
    //ncnn推理对象
    QScopedPointer<ncnn::Net> m_net{};
    //模型文件
    QString m_modelFile;
    //模型参数文件
    QString m_paramFile;
    //模型输入标识
    QString m_inputName;
    //模型输出标识
    QString m_outputName;
    //模型可识别宽高
    int m_modelW;
    int m_modelH;
    //模型初始化状态
    bool m_isInit = false;
};

#endif // QTNCNNMODNET_H
