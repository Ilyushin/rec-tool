#!/usr/bin/env bash


rm -rf ./cf_experiments_loop.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
pip3 uninstall -y cf_experiments_loop
pip3 install ./dist/cf_experiments_loop-*-py3-none-any.whl