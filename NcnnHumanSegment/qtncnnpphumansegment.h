#ifndef QTNCNNPPHUMANSEGMENT_H
#define QTNCNNPPHUMANSEGMENT_H

#include <QObject>
#include <QMutex>
#include <QImage>

namespace ncnn {
class Net;
}
// 使用PP-Human-Segment模型分割人物与背景
class QtNcnnPPHumanSegment : public QObject
{
    Q_OBJECT
public:
    explicit QtNcnnPPHumanSegment(QObject *parent,
        const QString& paramFile = "simple_model_interp_192x192.param",
        const QString& modelFile = "simple_model_interp_192x192.bin",
        const QString& inputName = "x",
        const QString& outputName = "bilinear_interp_v2_13.tmp_0",
        int modelW = 192, int modelH = 192);
    explicit QtNcnnPPHumanSegment(
        const QString& paramFile = "simple_model_interp_192x192.param",
        const QString& modelFile = "simple_model_interp_192x192.bin",
        const QString& inputName = "x",
        const QString& outputName = "bilinear_interp_v2_13.tmp_0",
        int modelW = 192, int modelH = 192, QObject *parent = nullptr);
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

#endif // QTNCNNPPHUMANSEGMENT_H
