#include "qtncnnpphumansegment.h"
#include <net.h>

static const float g_sPPHumanSegNormVals[3] = { 1.0f / (0.5f * 255), 1.0f / (0.5f * 255), 1.0f / (0.5f * 255) };
static const float g_sPPHumanSegMeanVals[3] = { 0.5f * 255, 0.5f * 255, 0.5f * 255 };

QtNcnnPPHumanSegment::QtNcnnPPHumanSegment(QObject *parent,
    const QString &paramFile, const QString &modelFile,
    const QString &inputName, const QString &outputName,
    int modelW, int modelH) : QtNcnnPPHumanSegment(paramFile, modelFile,
    inputName, outputName, modelW, modelH, parent) {}

QtNcnnPPHumanSegment::QtNcnnPPHumanSegment(
    const QString &paramFile, const QString &modelFile,
    const QString &inputName, const QString &outputName,
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

QImage QtNcnnPPHumanSegment::segmentImage(QImage imgBgra, QImage* retMask)
{
    if (!init())
        return QImage();
    //QImage ret;

    //转换色彩空间
    QImage rgbImage = imgBgra.convertToFormat(QImage::Format_RGB888);
    rgbImage = rgbImage.rgbSwapped();
    //转换图像尺寸
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbImage.bits(),
        ncnn::Mat::PIXEL_RGB, rgbImage.width(), rgbImage.height(), m_modelW, m_modelH);
    in.substract_mean_normalize(g_sPPHumanSegMeanVals, g_sPPHumanSegNormVals);
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
    const float* p0 = out.channel(0);
    const float* p1 = out.channel(1);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            uchar *curData = mask.scanLine(y) + x;
            const int idx = y * out.w + x;
            float b = p0[idx];  // backgroud
            float h = p1[idx];  // human
            float prob = qExp(h) / (qExp(b) + qExp(h) + 1e-6f);
            *curData = static_cast<uchar>(prob * 255);
        }
    }
    mask = mask.scaled(imgBgra.size());
    if(retMask)
        *retMask = mask;
    QImage ret = imgBgra.convertToFormat(QImage::Format_ARGB32);
    ret.setAlphaChannel(mask);

    return ret;
}

bool QtNcnnPPHumanSegment::init()
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
