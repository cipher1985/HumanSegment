/********************************************************
*
* 文件名：     qtcamera.h
* 版权：       ChinaEdu Co. Ltd. Copyright 2024 All Rights Reserved.
* 描述：       基于QT的摄像头采集类
*
* 修改人：     傅祯勇
* 修改内容：
* 版本：       1.0
* 修改时间：   2024-07-25
*
********************************************************/
#ifndef QTCAMERACAPTURE_H
#define QTCAMERACAPTURE_H

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QEventLoop>
#include <QAbstractVideoSurface>
#include <QCamera>

class QCameraImageCapture;
class QCameraInfo;
//基于QT的摄像头采集类
class QtCameraCapture : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    struct Info{
        QString id;
        QString name;
        QList<QSize> m_resolutions;
    };
    static QList<Info> infoList(bool getResolutions = false);
public:
    explicit QtCameraCapture(QObject *parent = nullptr);
    //打开摄像头
    bool open(int index = 0);
    bool open(const QString& id);
    bool isOpen();
    //关闭摄像头
    void close();
    //摄像头支持的分辨率
    QList<QSize> supportedResolutions();
    //设置分辨率
    void setResolution(const QSize& size);
    void setResolution(int w, int h);
    //主动采集图像
    QImage captureImage();
Q_SIGNALS:
    //摄像头开始采集
    void sigStartCaptured();
    //摄像头采集异常断开
    void sigDisconnect();
    //自动回调采集图像
    void sigCaptureFrame(QImage image);
private:
    //获得渲染回调数据
    virtual bool start(const QVideoSurfaceFormat &format) Q_DECL_OVERRIDE;
    virtual QList<QVideoFrame::PixelFormat> supportedPixelFormats(
        QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const Q_DECL_OVERRIDE;
    virtual bool isFormatSupported(const QVideoSurfaceFormat &format) const Q_DECL_OVERRIDE;
    virtual bool present(const QVideoFrame &frame) Q_DECL_OVERRIDE;
private:
    bool open(const QCameraInfo* info);
    QRecursiveMutex m_mutex;
    QEventLoop m_loopEvent;
    QImage m_image;
    QCamera* m_camera{};
    QCameraImageCapture* m_imageCapture{};
    QList<QSize> m_resolutions;
};

#endif // QTCAMERACAPTURE_H
