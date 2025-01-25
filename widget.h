#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QElapsedTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class QtCameraCapture;
class QtOnnxRobustVideoMatting;
class QtOnnxModNet;
class QtNcnnModNet;
class QLabel;
class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
private:
    void setBgColor(const QColor& color);
    void updateSegment();
    void updateFps(const QString& fps = QString());
private slots:
    void on_pushButton_refresh_clicked();
    void on_pushButton_capture_clicked();
    void on_checkBox_reverse_x_stateChanged(int state);
    void on_checkBox_reverse_y_stateChanged(int state);
    void on_pushButton_process_clicked();
    void on_pushButton_set_color_clicked();
    void on_pushButton_set_bg_clicked();
    void on_comboBox_mode_list_currentIndexChanged(int index);
    void on_pushButton_save_clicked();
private:
    Ui::Widget *ui;
    QElapsedTimer m_timer;
    QAtomicInt m_modeIndex{0};
    QImage m_imgFG;
    QImage m_imgBG;
    QColor m_bgColor{153, 255, 120, 255};
    QLabel* m_labelFG{};
    QLabel* m_labelBG{};
    QtCameraCapture* m_cam{};
    QtOnnxRobustVideoMatting* m_onnxRvm{};
    QtOnnxModNet* m_onnxModNet{};
    QtNcnnModNet* m_ncnnModNet{};
    int m_frameCount{0};
    std::atomic_bool m_reverseX{false};
    std::atomic_bool m_reverseY{false};
};
#endif // WIDGET_H
