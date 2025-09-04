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
* 版本：       1.1
* 修改时间：   2025-08-07
*
********************************************************/

#ifndef QTCAMERACAPTURE_H
#define QTCAMERACAPTURE_H

#include <QObject>
#include <QImage>
#include <QMediaCaptureSession>
#include <QVideoSink>
#include <QCamera>
#include <QCameraDevice>
#include <QMutex>
#include <QElapsedTimer>

class QtCameraCapture : public QObject
{
    Q_OBJECT
public:
    struct Info {
        QString id;
        QString name;
        QList<QSize> resolutions;
    };
    static QList<Info> infoList(bool getResolutions = false);

    explicit QtCameraCapture(QObject *parent = nullptr);
    ~QtCameraCapture();
    // 采集状态
    bool open(int index = 0);
    bool open(const QString &id);
    bool isOpen() const;
    void close();
    // 获得支持分辨率
    QList<QSize> supportedResolutions() const;
    void setResolution(const QSize &size);
    void setResolution(int w, int h);
    // 获得当前图像
    QImage captureImage();
signals:
    void sigCaptureFrame(const QImage &img);
private:
    mutable QMutex          m_mtx;
    QCamera*                m_camera     = nullptr;
    QMediaCaptureSession    m_session;
    QVideoSink*             m_sink       = nullptr;
    QImage                  m_lastFrame;
    QList<QSize>            m_resolutions;
    QElapsedTimer           m_elapsedTimer;
};

#endif // QTCAMERACAPTURE_H
