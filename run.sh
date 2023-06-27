#!/bin/bash

python3 -m ensurepip --default-pip

pip install -r requirements.txt

matlab GSML_ADSAILV2.m