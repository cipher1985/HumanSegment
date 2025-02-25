﻿#ifndef QTONNXMODNET_H
#define QTONNXMODNET_H

#include <QObject>
#include <QImage>

#include <onnxruntime_cxx_api.h>

//ncnn使用ModNet模型分割人物与背景
class QtOnnxModNet : public QObject
{
    Q_OBJECT
public:
    explicit QtOnnxModNet(
        const QString& modelFile = "modnet_webcam_portrait_matting_256x256.onnx",
        int modelW = 256, int modelH = 256,
        int opThreads = 0, QObject *parent = nullptr);
    explicit QtOnnxModNet(QObject *parent,
        const QString& modelFile = "modnet_webcam_portrait_matting_256x256.onnx",
        int modelW = 256, int modelH = 256,
        int opThreads = 0);
    // 获得分割图像
    QImage segmentImage(QImage imgBgra, QImage* retMask = nullptr);
private:
    // 将图像数据转换为ONNX Runtime可处理的格式
    std::vector<Ort::Value> transformImageData(QImage imgBgr);
    // 图像数据归一化处理
    std::vector<float> normalizeImageData(QImage img);
    // 生成分割通道图像
    QImage generateMatting(std::vector<Ort::Value> &outputTensors);
private:
    // ONNX Runtime环境对象
    Ort::Env m_env;
    // ONNX Runtime会话选项对象
    Ort::SessionOptions m_sessionOptions;
    // ONNX Runtime会话对象指针
    QScopedPointer<Ort::Session> m_session{};
    // 输入输出节点名称
    std::string m_inputName;
    std::string m_outputName;
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;
    // 输入节点维度{b=1,c,h,w}
    std::vector<int64_t> m_inputNodeDims;
    // 归一化输入数据
    std::vector<float> m_normalizeInputData;
};

#endif // QTONNXMODNET_H
