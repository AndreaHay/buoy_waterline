#! /bin/bash

# combine csv of results from each scale

echo "ID,mean,median,stdev,std_err,Q1,Q3,IQR,no_waterlines_detected,no_images_total" >/working/projects/buoy_waterline/results/BUOY_WATERLINE_RESULTS.csv

for D in /working/projects/buoy_waterline/*; do
	if [ -d "${D}" ]; then
		awk 'FNR==2' "${D}"/output/*_final* >> /working/projects/buoy_waterline/results/BUOY_WATERLINE_RESULTS.csv
	fi
done

