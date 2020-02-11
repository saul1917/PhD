#!/bin/bash
echo 'Executing PSP Plate classification tests'
echo 'Tests for InceptionV3'
python mainTests.py --arch "inception_v3"
echo 'Tests for Resnet18'
python mainTests.py --arch "resnet18"
echo 'Tests for Resnet50'
python mainTests.py --arch "resnet50"

