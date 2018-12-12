#!/bin/bash
cd ./flow_networks/correlation_package
python3 setup.py install
cd ../resample2d_package 
python3 setup.py install
cd ../channelnorm_package 
python3 setup.py install
cd ..
