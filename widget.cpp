#include "widget.h"
#include "ui_widget.h"

#include <QPainter>
#include <QMessageBox>
#include <QColorDialog>
#include <QFileDialog>
#include <QStackedLayout>
#include <QStyledItemDelegate>


#include "qtcameracapture.h"
#include "qtncnnmodnet.h"
#include "qtncnnpphumansegment.h"
#include "qtonnxmodnet.h"
#include "qtonnxrobustvideomatting.h"
#include "qtonnxpphumansegment.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    ui->comboBox_cam_list->setItemDelegate(new QStyledItemDelegate);
    ui->comboBox_mode_list->setItemDelegate(new QStyledItemDelegate);

    QString onnxRvm = m_tempDir.filePath("rvm.onnx");
    QFile::copy("://OnnxHumanSegment/model/rvm/rvm_mobilenetv3_fp32.onnx", onnxRvm);
    QString onnxPPHumanSeg = m_tempDir.filePath("ppHumanSeg.onnx");
    QFile::copy("://OnnxHumanSegment/model/PP-HumanSeg/model_interp.onnx", onnxPPHumanSeg);
    QString onnxModnet = m_tempDir.filePath("modnet.onnx");
    QFile::copy("://OnnxHumanSegment/model/modnet/modnet_webcam_portrait_matting_256x256.onnx", onnxModnet);
    QString ncnnModnetBin = m_tempDir.filePath("modnet.bin");
    QString ncnnModnetParam = m_tempDir.filePath("modnet.param");
    QFile::copy("://NcnnHumanSegment/model/modnet/modnet_webcam_portrait_matting_256x256.bin", ncnnModnetBin);
    QFile::copy("://NcnnHumanSegment/model/modnet/modnet_webcam_portrait_matting_256x256.param", ncnnModnetParam);
    QString ncnnPPHumanSegBin = m_tempDir.filePath("ppHumanSeg.bin");
    QString ncnnPPHumanSegParam = m_tempDir.filePath("ppHumanSeg.param");
    QFile::copy("://NcnnHumanSegment/model/PP-HumanSeg/simple_model_interp_192x192.bin", ncnnPPHumanSegBin);
    QFile::copy("://NcnnHumanSegment/model/PP-HumanSeg/simple_model_interp_192x192.param", ncnnPPHumanSegParam);

    m_onnxRvm = new QtOnnxRobustVideoMatting(this, onnxRvm);
    m_onnxPPHumanSeg = new QtOnnxPPHumanSegment(this, onnxPPHumanSeg);
    m_onnxModNet = new QtOnnxModNet(this, onnxModnet);
    m_ncnnModNet = new QtNcnnModNet(this, ncnnModnetParam, ncnnModnetBin);
    m_ncnnPPHumanSeg = new QtNcnnPPHumanSegment(this, ncnnPPHumanSegParam, ncnnPPHumanSegBin);

    m_cam = new QtCameraCapture(this);

    QStackedLayout* layout = new QStackedLayout(ui->frame);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setStackingMode(QStackedLayout::StackAll);
    m_labelFG = new QLabel();
    m_labelFG->setScaledContents(true);
    m_labelBG = new QLabel();
    m_labelBG->setScaledContents(true);
    layout->addWidget(m_labelFG);
    layout->addWidget(m_labelBG);

    connect(m_cam, &QtCameraCapture::sigCaptureFrame,
        this, [this](QImage img) {
        img = img.mirrored(m_reverseX, !m_reverseY);
        m_imgFG = img;
        updateSegment();
        updateFps();
    }, Qt::QueuedConnection);
    on_pushButton_refresh_clicked();
    setBgColor(m_bgColor);
}

Widget::~Widget()
{
    delete ui;
}


void Widget::setBgColor(const QColor &color)
{
    m_bgColor = color;
    QString colorStyle = QString("background: rgba(%1, %2, %3, %4);")
        .arg(m_bgColor.red()).arg(m_bgColor.green())
        .arg(m_bgColor.blue()).arg(m_bgColor.alpha());
    ui->label_color->setStyleSheet(colorStyle);
    ui->label_color->setPixmap(QPixmap());
    m_labelBG->setStyleSheet(colorStyle);
    m_labelBG->setPixmap(QPixmap());
}

void Widget::updateSegment()
{
    QImage imgFG;
    switch (m_modeIndex) {
    case 1://ONNX RVM
        imgFG = m_onnxRvm->segmentImage(m_imgFG);
        break;
    case 2://ONNX MODNet
        imgFG = m_onnxModNet->segmentImage(m_imgFG);
        break;
    case 3://ONNX PP-HumanSegment
        imgFG = m_onnxPPHumanSeg->segmentImage(m_imgFG);
        break;
    case 4://NCNN MODNet
        imgFG = m_ncnnModNet->segmentImage(m_imgFG);
        break;
    case 5://NCNN PP-HumanSegment
        imgFG = m_ncnnPPHumanSeg->segmentImage(m_imgFG);
        break;
    default:
        //ONNX RVM
        imgFG = m_imgFG;
        break;
    }
    m_labelFG->setPixmap(QPixmap::fromImage(imgFG));
}

void Widget::updateFps(const QString& fps)
{
    if(!fps.isEmpty()) {
        ui->label_fps->setText(fps);
        return;
    }
    m_frameCount++;
    int elapsed = m_timer.elapsed();
    if (elapsed >= 1000) {
        float fps = m_frameCount / (elapsed / 1000.0f);
        m_timer.restart();
        m_frameCount = 0;
        ui->label_fps->setText(QString("%1").arg(fps, 0, 'f', 2));
    }
}

void Widget::on_pushButton_refresh_clicked()
{
    //刷新摄像头列表
    QList<QtCameraCapture::Info> info = QtCameraCapture::infoList();
    ui->comboBox_cam_list->clear();
    for (auto& i : info)
        ui->comboBox_cam_list->addItem(i.name);
    ui->comboBox_cam_list->setCurrentIndex(0);
}

void Widget::on_pushButton_capture_clicked()
{
    int index = ui->comboBox_cam_list->currentIndex();
    if(!m_cam->open(index)) {
        QMessageBox::information(this, u8"提示", u8"摄像头打开失败", QMessageBox::Ok);
        return;
    }
    m_imgFG = QImage();
    m_cam->setResolution(320, 240);
    m_timer.restart();
    m_frameCount = 0;
}
void Widget::on_checkBox_reverse_x_stateChanged(int state)
{
    m_reverseX = (state == Qt::Checked);
}

void Widget::on_checkBox_reverse_y_stateChanged(int state)
{
    m_reverseY = (state == Qt::Checked);
}

void Widget::on_pushButton_process_clicked()
{
    //选择抠像图片
    QFileDialog dlg(this);
    dlg.setWindowTitle(u8"选择抠像图片");
    dlg.setFileMode(QFileDialog::ExistingFile);
    dlg.setAcceptMode(QFileDialog::AcceptOpen);
    dlg.setNameFilter(u8"抠像图片 (*.jpg *.bmp *.png *.jpeg)");
    if (dlg.exec() != QDialog::Accepted)
        return;
    QString imgFile = dlg.selectedFiles().at(0);
    QImage img;
    if(!img.load(imgFile))
        return;
    m_cam->close();
    m_imgFG = img;
    updateSegment();
    updateFps("0.00");
}

void Widget::on_pushButton_set_color_clicked()
{
    //设置背景颜色
    QColorDialog dlg(this);
    dlg.setWindowTitle(u8"选择背景颜色");
    dlg.setCurrentColor(m_bgColor);
    dlg.setOptions(QColorDialog::ShowAlphaChannel);
    if (dlg.exec() != QDialog::Accepted)
        return;
    setBgColor(dlg.currentColor());
}

void Widget::on_pushButton_set_bg_clicked()
{
    //设置背景图片
    QFileDialog dlg(this);
    dlg.setWindowTitle(u8"选择背景图片");
    dlg.setFileMode(QFileDialog::ExistingFile);
    dlg.setAcceptMode(QFileDialog::AcceptOpen);
    dlg.setNameFilter(u8"背景图片 (*.jpg *.bmp *.png *.jpeg)");
    if (dlg.exec() != QDialog::Accepted)
        return;
    QString imgFile = dlg.selectedFiles().at(0);
    QImage img;
    if(!img.load(imgFile))
        return;
    m_imgBG = img;
    ui->label_color->setStyleSheet("background: transparne;");
    QPixmap pixmap = QPixmap::fromImage(m_imgBG);
    ui->label_color->setPixmap(pixmap);
    m_labelBG->setPixmap(pixmap);
}

void Widget::on_comboBox_mode_list_currentIndexChanged(int index)
{
    m_modeIndex = index;
    if(m_cam->isOpen() || m_imgFG.isNull())
        return;
    updateSegment();
    updateFps("0.00");
}

void Widget::on_pushButton_save_clicked()
{
#if QT_VERSION_MAJOR >= 6
    QPixmap pixFg = m_labelFG->pixmap().copy();
#else
    QPixmap pixFg = m_labelFG->pixmap()->copy();
#endif
    if(pixFg.isNull())
        return;
#if QT_VERSION_MAJOR >= 6
    QPixmap pixBg = m_labelBG->pixmap().copy();
#else
    QPixmap pixBg = m_labelBG->pixmap()->copy();
#endif
    QPixmap pixmap(pixFg.width(), pixFg.height());
    pixmap.fill(m_bgColor);
    QPainter painter(&pixmap);
    //绘制背景
    if(!pixBg.isNull())
        painter.drawPixmap(0, 0, pixFg.width(), pixFg.height(), pixBg);
    //绘制前景
    painter.drawPixmap(0, 0, pixFg);
    //保存抠像图片
    QFileDialog dlg(this);
    dlg.setWindowTitle(u8"保存图片");
    dlg.setFileMode(QFileDialog::AnyFile);
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setNameFilter(u8"保存图片 (*.jpg *.bmp *.png *.jpeg)");
    if (dlg.exec() != QDialog::Accepted)
        return;
    QString imgFile = dlg.selectedFiles().at(0);
    QString extFile = QFileInfo(imgFile).suffix().toLower();
    if(extFile != "jpg" && extFile != "bmp" &&
        extFile != "png" && extFile != "jpeg")
        imgFile.append(".png");
    if(!pixmap.save(imgFile)) {
        QMessageBox::information(this, u8"提示", u8"图片保存失败", QMessageBox::Ok);
        return;
    }
    QMessageBox::information(this, u8"提示", u8"图片保存完成", QMessageBox::Ok);
}
