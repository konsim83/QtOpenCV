#ifndef OPENCVWORKER_H
#define OPENCVWORKER_H

#include <QObject>
#include <QImage>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class OpenCvWorker : public QObject
{
    Q_OBJECT

private:
    cv::Mat _frameOriginal;
    cv::Mat _frameProcessed;
    cv::VideoCapture *cap;

    bool status;
    bool toggleStream;

    bool binaryThresholdEnable;
    int binaryThreshold;

    void checkIfDeviceAlreadyOpened(const int device);
    void process();

public:
    explicit OpenCvWorker(QObject *parent = nullptr);
    ~OpenCvWorker();

signals:
    void sendFrame(QImage frameProcessed);
//    void sendStatus(QString msg, int code);

public slots:
    void receiveGrabFrame();
    void receiveSetup(const int device);
    void receiveToggleStream();

    void receiveEnableBinaryThreshold();
    void receiveBinaryThreshold(int threshold);
};

#endif // OPENCVWORKER_H
