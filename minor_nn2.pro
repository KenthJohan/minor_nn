TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += minor_nn2.c

HEADERS += lin.h
HEADERS += mnn.h

DEFINES += __USE_MINGW_ANSI_STDIO
