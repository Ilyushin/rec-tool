#!/usr/bin/env bash


rm -rf ./rec_tool.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
pip3 uninstall -y rec_tool
pip3 install ./dist/rec_tool-*-py3-none-any.whl