#!/bin/bash

SkipTransform="false"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

matlab -nodisplay -nosplash -r "try, addpath('src'); main($SkipTransform); catch ex, disp(getReport(ex, 'extended')), exit(1), end, exit(0)"
# matlab -nodisplay -nosplash -r "addpath('src'); main($SkipTransform);"
# matlab -r "addpath('src'); run('src/main.m')"