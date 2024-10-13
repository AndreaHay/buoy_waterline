#! /bin/bash

#############################################################
# Workflow script to process multiple scales
#
# A. Hay - 210607
############################################################

# scales to run (uncomment and edit this instead of giving flags if you like)
#scales=(9A 9B 9C)
echo 'processing buoy waterlines'

while getopts d:s:h flag
do
  case "${flag}" in
    d) images_dir=${OPTARG};;
    s) scales+=("$OPTARG");;
    h)
      echo -e "\n SETTINGS:"
      echo "images directory (-d) (USE FULL DIR PATH, e.g. /working/projects/buoy_waterline/)"
      echo "scales (-s) (each scale, e.g. -s 8A -s 10B -s 8C (etc.). These must be the names of subdirs, containing .MOV files)"
      exit 0
  esac
done


# for each scale
for scale in "${scales[@]}"; do
	echo "scale: $scale"
	buoy_float_level.sh -d ${images_dir}${scale}/ -n ${scale} -r 5 -e t -o 1 >run_${scale}.log 2>&1 &
done



