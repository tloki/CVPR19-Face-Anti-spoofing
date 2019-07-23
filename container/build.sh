#!/usr/bin/env bash

#T_ODO: move to the path where this script is located first!

#cp ../requirements.txt .

docker build . -t antispoof

#rm requirements.txt