#ifndef QTONNXROBUSTVIDEOMATTING_H
#define QTONNXROBUSTVIDEOMATTING_H

#include <QObject>
#include <QImage>

#include <onnxruntime_cxx_api.h>

//ncnn使用RVM模型分割人物与背景
class QtOnnxRobustVideoMatting : public QObject
{
    Q_OBJECT
public:
    explicit QtOnnxRobustVideoMatting(
        const QString& modelFile = "rvm_mobilenetv3_fp32.onnx",
        int opThreads = 0, QObject* parent = nullptr);
    explicit QtOnnxRobustVideoMatting(QObject* parent,
        const QString& modelFile = "rvm_mobilenetv3_fp32.onnx",
        int opThreads = 0);
    // 获得分割图像
    QImage segmentImage(QImage imgBgra, QImage* retMask = nullptr);
private:
    // 将图像数据转换为ONNX Runtime可处理的格式
    std::vector<Ort::Value> transformImageData(QImage imgBgr);
    // 图像数据归一化处理
    QVector<float> normalizeImageData(QImage img);
    // 计算张量维度对应的元素数量(数据内容相乘)
    int64_t valueSizeOf(const std::vector<int64_t> &dims);
    // 生成分割通道图像
    QImage generateMatting(std::vector<Ort::Value> &outputTensors);
    // 更新模型信息准备下次处理
    void updateContext(std::vector<Ort::Value> &outputTensors);
private:
    // ONNX Runtime 的环境对象
    Ort::Env m_env;
    // ONNX Runtime 的会话选项对象
    Ort::SessionOptions m_sessionOptions;
    // ONNX Runtime 的会话对象指针
    QScopedPointer<Ort::Session> m_session{};
    // 模型节点数量
    static const unsigned int m_nodeNum = 6;
    // 模型输入节点名称
    const char * input_node_names[m_nodeNum]{
        "src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"};
    // 模型输出节点名称
    const char * output_node_names[m_nodeNum]{
        "fgr", "pha", "r1o", "r2o", "r3o", "r4o"};
    // 输入节点维度
    std::vector<int64_t> m_dynamicInputNodeDims[m_nodeNum] = {
        {1, 3, 3, 4}, // src(b=1,c,h,w)
        {1, 1, 1, 1}, // r1i
        {1, 1, 1, 1}, // r2i
        {1, 1, 1, 1}, // r3i
        {1, 1, 1, 1}, // r4i
        {1} // downSampleRatio dst
    }; // (1, 16, ?h, ?w) for inner loop rxi
    // 归一化输入数据
    QVector<float> m_normalizeInputData;//dynamic_src_value_handler;
    // 动态处理存储中间结果
    std::vector<float> m_r1iDynamicValueHandler = { 0.0f }; // init 0. with shape (1,1,1,1)
    std::vector<float> m_r2iDynamicValueHandler = { 0.0f };
    std::vector<float> m_r3iDynamicValueHandler = { 0.0f };
    std::vector<float> m_r4iDynamicValueHandler = { 0.0f };
    std::vector<float> m_dstDynamicValueHandler = { 0.25f }; // downsampleRatio with shape (1)
};

#endif // QTONNXROBUSTVIDEOMATTING_H
