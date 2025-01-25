#include "qtncnnmodnet.h"
#include <net.h>

static const float g_sModnetNormVals[3] = { 1 / 177.5f, 1 / 177.5f, 1 / 177.5f };
static const float g_sModnetMeanVals[3] = { 175.5f, 175.5f, 175.5f };

QtNcnnModNet::QtNcnnModNet(QObject *parent,
    const QString& paramFile, const QString& modelFile,
    const QString& inputName, const QString& outputName,
    int modelW, int modelH): QtNcnnModNet(paramFile, modelFile,
        inputName, outputName, modelW, modelH, parent) {}

QtNcnnModNet::QtNcnnModNet(
    const QString& paramFile, const QString& modelFile,
    const QString& inputName, const QString& outputName,
    int modelW, int modelH, QObject *parent)
    : QObject{parent}
{
    m_net.reset(new ncnn::Net);

    m_inputName = inputName;
    m_outputName = outputName;

    m_modelW = modelW;
    m_modelH = modelH;

    m_modelFile = modelFile;
    m_paramFile = paramFile;

    init();
}

QImage QtNcnnModNet::segmentImage(QImage imgBgra, QImage* retMask)
{
    if (!init())
        return QImage();
    //转换色彩空间
    QImage rgbImage = imgBgra.convertToFormat(QImage::Format_RGB888);
    rgbImage = rgbImage.rgbSwapped();
    //转换图像尺寸
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbImage.bits(),
        ncnn::Mat::PIXEL_RGB, rgbImage.width(), rgbImage.height(), m_modelW, m_modelH);
    in.substract_mean_normalize(g_sModnetMeanVals, g_sModnetNormVals);
    //创建拾取器
    ncnn::Extractor ex = m_net->create_extractor();
    //ex.set_num_threads(1);
    ex.input(m_inputName.toStdString().c_str(), in);
    ncnn::Mat out;//0-1浮点型透明度
    ex.extract(m_outputName.toStdString().c_str(), out);
    if(out.data == nullptr)
        return QImage();
    //创建蒙版
    QImage mask(out.h, out.w, QImage::Format_Grayscale8);
    const float* probMap = out.channel(0);
    for (int i = 0; i < out.h; ++i) {
        for (int j = 0; j < out.w; ++j) {
            uchar *curData = mask.scanLine(i) + j;
            *curData = (uchar)(probMap[i * out.w + j] * 255);
        }
    }
    mask = mask.scaled(imgBgra.size());
    if(retMask)
        *retMask = mask;
    QImage ret = imgBgra.convertToFormat(QImage::Format_ARGB32);
    ret.setAlphaChannel(mask);
    return ret;
}

bool QtNcnnModNet::init()
{
    if(m_isInit)
        return true;
    //初始化组件
    m_net->clear();
    //设置gpu模式
    m_net->opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
    //fp16运算加速
    //m_net->opt.use_fp16_arithmetic = true;
    if (m_net->load_param(m_paramFile.toStdString().c_str()) != 0 ||
        m_net->load_model(m_modelFile.toStdString().c_str()) != 0) {
        m_net->clear();
        return false;
    }
    m_isInit = true;
    return true;
}
