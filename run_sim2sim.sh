#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Error: Please provide robot namespace"
    echo "Example: ./run_sim2sim.sh TRON1"
    exit 1
fi

robot_ns=$1

python3 "scripts/$robot_ns/sim2sim.py"
