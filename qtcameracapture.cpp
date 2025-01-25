#include "qtcameracapture.h"

//#include <QCamera>
#include <QCameraInfo>
#include <QCameraImageCapture>
#include <QVideoSurfaceFormat>
//#include <QDebug>

QList<QtCameraCapture::Info> QtCameraCapture::infoList(bool getResolutions)
{
    QList<QCameraInfo> list = QCameraInfo::availableCameras();
    int count = list.count();
    QList<QtCameraCapture::Info> ret;
    for (int i = 0; i < count; ++i) {
        QtCameraCapture::Info curCamera;
        QCameraInfo& info = list[i];
        curCamera.id = info.deviceName();
        curCamera.name = info.description();
        if(!getResolutions) {
            ret.append(curCamera);
            continue;
        }
        //获得分辨率
        QCamera camera(info);
        camera.load();
        curCamera.m_resolutions = camera.supportedViewfinderResolutions();
        camera.unload();
        ret.append(curCamera);
    }
    return ret;
}

QtCameraCapture::QtCameraCapture(QObject *parent)
    : QAbstractVideoSurface{parent}
{
}

bool QtCameraCapture::open(int index)
{
    QList<QCameraInfo> list = QCameraInfo::availableCameras();
    if(index < 0 || index >= list.count())
        return false;
    QCameraInfo& info = list[index];
    return open(&info);
}

bool QtCameraCapture::open(const QString &deviceName)
{
    int index = -1;
    QList<QCameraInfo> list = QCameraInfo::availableCameras();
    for(int i = 0;i < list.count(); ++i) {
        if(list.at(i).deviceName() == deviceName) {
            index = i;
            break;
        }
    }
    if(index == -1)
        return false;
    QCameraInfo& info = list[index];
    return open(&info);
}

bool QtCameraCapture::isOpen()
{
    return (m_camera != nullptr);
}

void QtCameraCapture::close()
{
    QMutexLocker locker(&m_mutex);
    if(!m_camera)
        return;
    m_loopEvent.quit();
    //m_camera->stop();
    delete m_imageCapture;
    m_imageCapture = nullptr;
    delete m_camera;
    m_camera = nullptr;
    m_resolutions.clear();
    m_image = QImage();
}

QImage QtCameraCapture::captureImage()
{
    QMutexLocker locker(&m_mutex);
    if(!m_camera)
        return m_image;
    if(m_imageCapture->isReadyForCapture()) {
        //开始采集图像
        m_camera->searchAndLock();//半按快门
        m_imageCapture->capture();//按下快门
        m_camera->unlock();//抬起快门
        m_loopEvent.exec();
    }
    return m_image.copy();
}

void QtCameraCapture::setResolution(const QSize &size)
{
    QMutexLocker locker(&m_mutex);
    if(!m_camera)
        return;
    //查找所支持的最近分辨率
    int minDist = 0;
    QSize curSize;
    for(auto &i : m_resolutions) {
        int distX = i.width() - size.width();
        int distY = i.height() - size.height();
        int dist = distX * distX + distY * distY;
        if(curSize.isEmpty() || dist < minDist) {
            curSize = i;
            minDist = dist;
        }
    }
    QCameraViewfinderSettings setting;
    setting.setResolution(curSize);
    m_camera->setViewfinderSettings(setting);
}

void QtCameraCapture::setResolution(int w, int h)
{
    setResolution(QSize(w, h));
}

bool QtCameraCapture::start(const QVideoSurfaceFormat &videoformat)
{
    QImage::Format format = QVideoFrame::imageFormatFromPixelFormat(videoformat.pixelFormat());
    if(format != QImage::Format_Invalid &&
        !videoformat.frameSize().isEmpty()) {
        QAbstractVideoSurface::start(videoformat);
        return true;
    }
    return false;
}

QList<QVideoFrame::PixelFormat> QtCameraCapture::supportedPixelFormats(QAbstractVideoBuffer::HandleType type) const
{
    if(type == QAbstractVideoBuffer::NoHandle) {
        return QList<QVideoFrame::PixelFormat>()
            << QVideoFrame::Format_ARGB32
            << QVideoFrame::Format_ARGB32_Premultiplied
            << QVideoFrame::Format_RGB32
            << QVideoFrame::Format_RGB24
            << QVideoFrame::Format_RGB565
            << QVideoFrame::Format_RGB555
            << QVideoFrame::Format_ARGB8565_Premultiplied;
    }
    return QList<QVideoFrame::PixelFormat>();
}

bool QtCameraCapture::isFormatSupported(const QVideoSurfaceFormat &videoformat) const
{
    QImage::Format format = QVideoFrame::imageFormatFromPixelFormat(videoformat.pixelFormat());
    return (format != QImage::Format_Invalid);
}

bool QtCameraCapture::present(const QVideoFrame &frame)
{
    if (frame.isValid()){
        QImage img = frame.image().convertToFormat(QImage::Format_RGB32);
        m_image = img;
        emit sigCaptureFrame(img.copy());
        return true;
    }
    stop();
    return false;
}

bool QtCameraCapture::open(const QCameraInfo* info)
{
    QMutexLocker locker(&m_mutex);
    close();
    // 创建摄像头对象
    m_camera = new QCamera(*info, this);
    m_camera->setViewfinder(this);
    m_camera->setCaptureMode(QCamera::CaptureStillImage);
    //创建图像捕获对象
    m_imageCapture = new QCameraImageCapture(m_camera);
    m_imageCapture->setCaptureDestination(QCameraImageCapture::CaptureToBuffer);
    //采集图像信号处理
    connect(m_imageCapture,
        &QCameraImageCapture::imageCaptured, this,
        [this](int id, const QImage& previewImage){
        Q_UNUSED(id);
        m_image = previewImage.convertToFormat(QImage::Format_RGB32);
        m_loopEvent.quit();
    });
    //采集图像出错信号
    connect(m_imageCapture,
        QOverload<int, QCameraImageCapture::Error, const QString &>::of(&QCameraImageCapture::error), this,
        [this](int id, QCameraImageCapture::Error error, const QString &errorString) {
        //采集图像出错
        Q_UNUSED(id);
        Q_UNUSED(error);
        Q_UNUSED(errorString);
        //qDebug()<<id<<error<<errorString;
        m_loopEvent.quit();
    });
    //启动摄像头
    m_camera->start();
    //获取摄像头支持的分辨率、帧率等参数
    m_resolutions = m_camera->supportedViewfinderResolutions();
    //测试摄像头是否打开
    if (m_camera->availability() != QMultimedia::Available) {
        //摄像头打开失败
        close();
        return false;
    }
    //设置摄像头状态改变回调
    connect(m_camera, &QCamera::statusChanged, this, [this](QCamera::Status status){
        //qDebug() << "status:"<<status;
        if(status == QCamera::ActiveStatus) {
            emit sigStartCaptured();
        }else if(status == QCamera::UnloadedStatus) {
            close();
            emit sigDisconnect();
        }
    });
    return true;
}
