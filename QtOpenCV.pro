#-------------------------------------------------
#
# Project created by QtCreator 2020-03-22T16:45:01
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtOpenCV
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        src/main.cpp \
        src/qcvwidget.cpp \
        src/opencvworker.cpp

INCLUDEPATH += \
        $$PWD/includes/

HEADERS += \
        includes/qcvwidget.h \
        includes/opencvworker.h

FORMS += \
        ui/qcvwidget.ui

LIBS += \
        -lopencv_highgui \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_videoio \
        -lopencv_video

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
