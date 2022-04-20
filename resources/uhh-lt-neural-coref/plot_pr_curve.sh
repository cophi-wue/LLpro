#!/usr/bin/env bash
# Plots a ROC curve based on logged csv files
if [ $# -eq 0 ]; then
    echo "Usage $0 <input_csv_file> <output_png_file>"
    exit 1
fi

IN_FILE=$1
OUT_FILE=$2
gnuplot <<- EOF
    set xlabel "Recall"
    set ylabel "Precision"
    set title "Precision Recall-Curve"
    set grid
    set term png
    set output "${OUT_FILE}"
    set datafile separator ","
    plot "${IN_FILE}" using 2:1 with lines
EOF

