#include "qtonnxmodnet.h"

#include <thread>

QtOnnxModNet::QtOnnxModNet(QObject *parent, const QString &modelFile,
    int modelW, int modelH,
    int opThreads) : QtOnnxModNet(modelFile,
        modelW, modelH, opThreads, parent) {}

QtOnnxModNet::QtOnnxModNet(const QString &modelFile,
    int modelW, int modelH,
    int opThreads, QObject *parent) : QObject{parent},
    m_env(ORT_LOGGING_LEVEL_ERROR, "modnet")
{
    std::wstring model = modelFile.toStdWString();
    // 设置会话启用扩展图优化
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    // 设置多核CPU线程数
    if(opThreads < 1) {
        m_sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    } else {
        m_sessionOptions.SetIntraOpNumThreads(opThreads);
    }
    // 创建 ONNX Runtime 会话对象
    m_session.reset(new Ort::Session(m_env, model.c_str(), m_sessionOptions));

    // 获得模型输入输出字段名称
    Ort::AllocatorWithDefaultOptions allocator;
    // 获取输入名称
    Ort::AllocatedStringPtr input_name_Ptr =
        m_session->GetInputNameAllocated(0, allocator);
    m_inputName = input_name_Ptr.get();
    m_inputNodeNames = { m_inputName.c_str() };
    // 获得输出名称
    Ort::AllocatedStringPtr output_name_Ptr =
        m_session->GetOutputNameAllocated(0, allocator);
    m_outputName = output_name_Ptr.get();
    m_outputNodeNames = { m_outputName.c_str() };
    // 设置输入节点维度
    m_inputNodeDims = {1, 3, modelH, modelW};
}

QImage QtOnnxModNet::segmentImage(QImage imgBgra, QImage *retMask)
{
    QImage ret;
    if (imgBgra.isNull())
        return ret;
    // 转换色彩空间
    QImage rgbImage = imgBgra.convertToFormat(QImage::Format_RGB888);
    // 将图像数据转换为ONNX Runtime可处理的格式
    std::vector<Ort::Value> input_tensors_ = transformImageData(rgbImage);
    // 运行ONNX Runtime会话进行推理处理
    std::vector<Ort::Value> outputTensors = m_session->Run(
        Ort::RunOptions{ nullptr },
        m_inputNodeNames.data(),
        input_tensors_.data(),
        m_inputNodeNames.size(),
        m_outputNodeNames.data(),
        m_outputNodeNames.size()
    );
    // 根据输出数据转换为分割遮罩图像
    QImage mask = generateMatting(outputTensors);
    mask = mask.scaled(imgBgra.width(), imgBgra.height());
    if(retMask)
        *retMask = mask;
    ret = imgBgra.convertToFormat(QImage::Format_ARGB32);
    ret.setAlphaChannel(mask);
    return ret;
}

QImage QtOnnxModNet::generateMatting(std::vector<Ort::Value> &outputTensors)
{
    //获得通道数据
    Ort::Value &alphaValue = outputTensors.at(0);
    //获得前景形状
    std::vector<int64_t> alphaDims =
        alphaValue.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    //输出高度
    int64_t h = alphaDims.at(2);
    //输出宽度
    int64_t w = alphaDims.at(3);
    //获得alpha数据
    float *probMap = alphaValue.GetTensorMutableData<float>();
    //创建蒙版
    QImage mask(w, h, QImage::Format_Grayscale8);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            uchar *curData = mask.scanLine(i) + j;
            *curData = (uchar)(probMap[i * w + j] * 255);
        }
    }
    return mask;
}

std::vector<Ort::Value> QtOnnxModNet::transformImageData(QImage imgBgr)
{
    //获得缩放图像
    int dstW = int(m_inputNodeDims[3]);
    int dstH = int(m_inputNodeDims[2]);
    QImage scaleImage = imgBgr.scaled(dstW, dstH);
    // 获得归一化数据
    m_normalizeInputData = normalizeImageData(scaleImage);
    // 创建内存信息
    Ort::MemoryInfo allocatorInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // 生成输入数据
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(allocatorInfo,
        m_normalizeInputData.data(), m_normalizeInputData.size(),
        m_inputNodeDims.data(), m_inputNodeDims.size()));

    return inputTensors;
}

std::vector<float> QtOnnxModNet::normalizeImageData(QImage img)
{
    // 定义图像的颜色通道数量
    const int colorSize = 3;
    int w = img.width();
    int h = img.height();
    int size = w * h;
    // 创建一个 QVector 用于存储归一化后的图像数据
    std::vector<float> output(size * colorSize);
    // 定义指向归一化后数据的指针数组
    float* normBits[3];
    normBits[0] = output.data();
    normBits[1] = normBits[0] + size;
    normBits[2] = normBits[1] + size;

    for (int y = 0; y < h; ++y) {
        const uchar* bits = img.constScanLine(y);
        int yOffset = y * w;
        int xOffset = 0;
        int bitXOffset = yOffset;
        for (int x = 0; x < w; ++x) {
            for(int c = 0; c < colorSize; ++c) {
                normBits[c][bitXOffset] = bits[xOffset] / 255.0f;
                ++xOffset;
            }
            ++bitXOffset;
        }
    }
    return output;
}
