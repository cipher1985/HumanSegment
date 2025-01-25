#include "qtonnxrobustvideomatting.h"

#include <thread>

QtOnnxRobustVideoMatting::QtOnnxRobustVideoMatting(
    QObject* parent, const QString& modelFile, int opThreads) :
        QtOnnxRobustVideoMatting(modelFile, opThreads, parent){}

QtOnnxRobustVideoMatting::QtOnnxRobustVideoMatting(
    const QString& modelFile, int opThreads, QObject* parent) :
    QObject(parent),
    m_env(ORT_LOGGING_LEVEL_ERROR, "robustvideomatting")
{
    std::wstring model = modelFile.toStdWString();
    // 设置会话选项，启用扩展图优化
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    // 设置多核CPU线程数
    if(opThreads < 1) {
        m_sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    } else {
        m_sessionOptions.SetIntraOpNumThreads(opThreads);
    }
    // 创建 ONNX Runtime 会话对象
    m_session.reset(new Ort::Session(m_env, model.c_str(), m_sessionOptions));
}

QImage QtOnnxRobustVideoMatting::segmentImage(QImage imgBgra, QImage* retMask)
{
    QImage ret;
    if (imgBgra.isNull())
        return ret;
    // 转换色彩空间
    QImage rgbImage = imgBgra.convertToFormat(QImage::Format_RGB888);
    // 生成缩放比让图像保持在512*512附近比例
    static const float keepScale = 512 * 512;
    static int lastSize = 0;
    static float lastDownSampleRatio = 0.25f;
    int size = imgBgra.width() * imgBgra.height();
    float downSampleRatio;
    if (size != lastSize) {
        downSampleRatio = keepScale / size;
        downSampleRatio = qMin(1.0f, downSampleRatio);
        lastSize = size;
        lastDownSampleRatio = downSampleRatio;
    } else {
        downSampleRatio = lastDownSampleRatio;
    }
    // 设置目标缩放比
    m_dstDynamicValueHandler[0] = downSampleRatio;
    // 将图像数据转换为ONNX Runtime可处理的格式
    std::vector<Ort::Value> inputTensors = transformImageData(rgbImage);
    // 运行ONNX Runtime会话进行推理处理
    std::vector<Ort::Value> outputTensors = m_session->Run(
        Ort::RunOptions{ nullptr }, input_node_names,
        inputTensors.data(), m_nodeNum, output_node_names, m_nodeNum);
    // 根据输出数据转换为分割遮罩图像
    QImage mask = generateMatting(outputTensors);
    if(retMask)
        *retMask = mask;
    ret = imgBgra.convertToFormat(QImage::Format_ARGB32);
    ret.setAlphaChannel(mask);
    // 更新初始化内容数据
    updateContext(outputTensors);
    return ret;
}

QVector<float> QtOnnxRobustVideoMatting::normalizeImageData(QImage img)
{
    // 定义图像的颜色通道数量
    const int colorSize = 3;
    int w = img.width();
    int h = img.height();
    int size = w * h;
    // 创建一个 QVector 用于存储归一化后的图像数据
    QVector<float> output(size * colorSize);
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

int64_t QtOnnxRobustVideoMatting::valueSizeOf(const std::vector<int64_t> &dims)
{
    if (dims.empty())
        return 0;
    int64_t ret = 1;
    for (const auto &size : dims)
        ret *= size;
    return ret;
}

std::vector<Ort::Value> QtOnnxRobustVideoMatting::transformImageData(QImage imgBgr)
{
    QImage src = imgBgr.copy();
    std::vector<int64_t> &srcDims = m_dynamicInputNodeDims[0]; // (1,3,h,w)
    srcDims[2] = imgBgr.height();
    srcDims[3] = imgBgr.width();

    // 获取其他输入节点的维度数组
    // assume that rxi's dims and value_handler was updated by last step in a while loop.
    std::vector<int64_t> &r1iDims = m_dynamicInputNodeDims[1]; // (1,?,?h,?w)
    std::vector<int64_t> &r2iDims = m_dynamicInputNodeDims[2]; // (1,?,?h,?w)
    std::vector<int64_t> &r3iDims = m_dynamicInputNodeDims[3]; // (1,?,?h,?w)
    std::vector<int64_t> &r4iDims = m_dynamicInputNodeDims[4]; // (1,?,?h,?w)
    std::vector<int64_t> &dsrDims = m_dynamicInputNodeDims[5]; // (1)

    // 计算其他输入节点的元素数量
    //int64_t srcValueSize = this->valueSizeOf(srcDims); // (1*3*h*w)
    int64_t r1iValueSize = this->valueSizeOf(r1iDims); // (1*?*?h*?w)
    int64_t r2iValueSize = this->valueSizeOf(r2iDims); // (1*?*?h*?w)
    int64_t r3iValueSize = this->valueSizeOf(r3iDims); // (1*?*?h*?w)
    int64_t r4iValueSize = this->valueSizeOf(r4iDims); // (1*?*?h*?w)
    int64_t dsrValueSize = this->valueSizeOf(dsrDims); // 1

    // 获得归一化数据
    m_normalizeInputData = normalizeImageData(src);
    // 创建内存信息
    Ort::MemoryInfo allocatorInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 生成输入数据
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_normalizeInputData.data(),
        m_normalizeInputData.count(), srcDims.data(), srcDims.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_r1iDynamicValueHandler.data(),
        r1iValueSize, r1iDims.data(), r1iDims.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_r2iDynamicValueHandler.data(),
        r2iValueSize, r2iDims.data(), r2iDims.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_r3iDynamicValueHandler.data(),
        r3iValueSize, r3iDims.data(), r3iDims.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_r4iDynamicValueHandler.data(),
        r4iValueSize, r4iDims.data(), r4iDims.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        allocatorInfo, m_dstDynamicValueHandler.data(),
        dsrValueSize, dsrDims.data(), dsrDims.size()));
    return inputTensors;
}

QImage QtOnnxRobustVideoMatting::generateMatting(std::vector<Ort::Value> &outputTensors)
{
    //获得通道数据
    Ort::Value &alphaValue = outputTensors.at(1); // pha (1,1,h,w) 0.~1.
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

void QtOnnxRobustVideoMatting::updateContext(std::vector<Ort::Value> &outputTensors)
{
    // 更新抠像数据信息0. update context for video matting.
    Ort::Value &r1o = outputTensors.at(2); // fgr (1,?,?h,?w)
    Ort::Value &r2o = outputTensors.at(3); // pha (1,?,?h,?w)
    Ort::Value &r3o = outputTensors.at(4); // pha (1,?,?h,?w)
    Ort::Value &r4o = outputTensors.at(5); // pha (1,?,?h,?w)
    std::vector<int64_t> r1oDims = r1o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> r2oDims = r2o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> r3oDims = r3o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> r4oDims = r4o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    // 根据上次输出rxo形状更新输入rxi形状
    m_dynamicInputNodeDims[1] = r1oDims;
    m_dynamicInputNodeDims[2] = r2oDims;
    m_dynamicInputNodeDims[3] = r3oDims;
    m_dynamicInputNodeDims[4] = r4oDims;
    // 根据上次输出rxo信息更新输入rxi信息
    int64_t new_r1i_value_size = this->valueSizeOf(r1oDims); // (1*?*?h*?w)
    int64_t new_r2i_value_size = this->valueSizeOf(r2oDims); // (1*?*?h*?w)
    int64_t new_r3i_value_size = this->valueSizeOf(r3oDims); // (1*?*?h*?w)
    int64_t new_r4i_value_size = this->valueSizeOf(r4oDims); // (1*?*?h*?w)
    m_r1iDynamicValueHandler.resize(new_r1i_value_size);
    m_r2iDynamicValueHandler.resize(new_r2i_value_size);
    m_r3iDynamicValueHandler.resize(new_r3i_value_size);
    m_r4iDynamicValueHandler.resize(new_r4i_value_size);
    float *new_r1i_value_ptr = r1o.GetTensorMutableData<float>();
    float *new_r2i_value_ptr = r2o.GetTensorMutableData<float>();
    float *new_r3i_value_ptr = r3o.GetTensorMutableData<float>();
    float *new_r4i_value_ptr = r4o.GetTensorMutableData<float>();
    // 更新动态数据信息
    std::memcpy(m_r1iDynamicValueHandler.data(), new_r1i_value_ptr, new_r1i_value_size * sizeof(float));
    std::memcpy(m_r2iDynamicValueHandler.data(), new_r2i_value_ptr, new_r2i_value_size * sizeof(float));
    std::memcpy(m_r3iDynamicValueHandler.data(), new_r3i_value_ptr, new_r3i_value_size * sizeof(float));
    std::memcpy(m_r4iDynamicValueHandler.data(), new_r4i_value_ptr, new_r4i_value_size * sizeof(float));
}

