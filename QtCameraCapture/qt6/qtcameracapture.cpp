#include "QtCameraCapture.h"
#include <QMediaDevices>

// ---------- static ----------
QList<QtCameraCapture::Info> QtCameraCapture::infoList(bool getResolutions)
{
    QList<QtCameraCapture::Info> ret;
    const auto cameras = QMediaDevices::videoInputs();
    for (const QCameraDevice &dev : cameras) {
        Info info;
        info.id   = dev.id();
        info.name = dev.description();
        if (getResolutions) {
            auto formats = dev.videoFormats();
            for(auto& i : formats) {
                info.resolutions.append(i.resolution());
            }
        }
        ret.append(info);
    }
    return ret;
}

// ---------- ctor / dtor ----------
QtCameraCapture::QtCameraCapture(QObject *parent)
    : QObject{parent}
{

}

QtCameraCapture::~QtCameraCapture()
{
    close();
}

// ---------- open / close ----------
bool QtCameraCapture::open(int index)
{
    const auto cameras = QMediaDevices::videoInputs();
    if (index < 0 || index >= cameras.size())
        return false;
    return open(cameras.at(index).id());
}

bool QtCameraCapture::open(const QString &id)
{
    close();

    const auto cameras = QMediaDevices::videoInputs();
    QCameraDevice selected;
    for (const auto &d : cameras) {
        if (d.id() == id) {
            selected = d;
            break;
        }
    }
    if (!selected.isNull())
        m_camera = new QCamera(selected, this);
    if (!m_camera)
        return false;

    m_sink = new QVideoSink(this);
    m_session.setCamera(m_camera);
    m_session.setVideoSink(m_sink);

    connect(m_sink, &QVideoSink::videoFrameChanged,
            this, [this](const QVideoFrame &frame){
        if (!frame.isValid()) return;
        if(m_elapsedTimer.elapsed() < 30)
            return;
        m_elapsedTimer.restart();
        QImage img = frame.toImage().convertToFormat(QImage::Format_RGB32);
        QMutexLocker l(&m_mtx);
        m_lastFrame = img;
        emit sigCaptureFrame(img);
    });

    // 分辨率列表
    m_resolutions.clear();
    for (const auto &fmt : selected.videoFormats())
        if (!m_resolutions.contains(fmt.resolution()))
            m_resolutions.append(fmt.resolution());

    m_camera->start();
    m_elapsedTimer.restart();
    return true;
}

bool QtCameraCapture::isOpen() const
{
    return m_camera && m_camera->isActive();
}

void QtCameraCapture::close()
{
    QMutexLocker l(&m_mtx);
    if (m_camera) {
        m_camera->stop();
        m_camera->deleteLater();
        m_camera = nullptr;
    }
    m_resolutions.clear();
    m_lastFrame = {};
}

// ---------- resolutions ----------
QList<QSize> QtCameraCapture::supportedResolutions() const
{
    return m_resolutions;
}

void QtCameraCapture::setResolution(const QSize &size)
{
    if (!m_camera)
        return;
    const auto dev = m_camera->cameraDevice();
    if (dev.isNull())
        return;
    // 所有支持的格式
    const auto fmts = dev.videoFormats();
    if (fmts.isEmpty())
        return;
    // 先求最小差距（宽差绝对值 + 高差绝对值）
    QCameraFormat format;
    int minDist = std::numeric_limits<int>::max();
    for (const QCameraFormat &f : fmts)
    {
        const QSize &r = f.resolution();
        int dist = qAbs(r.width()  - size.width()) +
                   qAbs(r.height() - size.height());
        if(minDist < dist)
            continue;
        minDist = dist;
        format = f;
    }
    if(format.isNull())
        return;
    m_camera->setCameraFormat(format);
}

void QtCameraCapture::setResolution(int w, int h)
{
    setResolution(QSize(w, h));
}

// ---------- capture ----------
QImage QtCameraCapture::captureImage()
{
    QMutexLocker l(&m_mtx);
    return m_lastFrame;
}

