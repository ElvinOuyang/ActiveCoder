#! /bin/bash

find . -type f -name *checkpoint* | xargs rm
find . -type d -name *pycache* | xargs rm -r