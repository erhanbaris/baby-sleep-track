#!/bin/sh

export QT_DEBUG_PLUGINS=1
export IN_DOCKER=1

python3 main.py
