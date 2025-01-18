#!/bin/bash

model_file="$1"
start=$2
step=$3
end=$4

if sed --version >/dev/null 2>&1; then
    sed_inplace="sed -i"  # Linux 系统
else
    sed_inplace="sed -i ''"  # MacOS 系统
fi

if [ ! -f "$model_file" ]; then
    echo "Error: File $model_file not found."
    exit 1
fi

for ((n=start; n<=end; n+=step)); do
    # $sed_inplace '$s/[[:space:]]*=[[:space:]]*[0-9]*/ = '"$i"'/' "$model_file"
    echo "Running with n = $n"
    python wfomc/enum_fo2_max.py -i "$model_file" -n "$n" -log C
done
