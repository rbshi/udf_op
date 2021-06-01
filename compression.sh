#!/bin/bash

FOLDERS=`ls`
for i in $FOLDERS
do
        if [[ $i == "build_"* ]]; then
                echo "Compressing $i"
                tar -czf $i.tar.gz $i
        fi
done

mv *.tar.gz ~/mnt_gdrive/gcp/ && cd ~/mnt_gdrive && grive -s gcp -l ~/mnt_gdrive/run.log $$ cd -