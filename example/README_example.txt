-------------------
TO RUN EXAMPLE
-------------------

1. Setup
- download all scripts and ensure all python packages are installed
- copy .MOV into new directory (we will call it '/waterline')
- set location of scale_coordinates.csv in buoy_single_image.sh and buoy_float_level.sh

2. Process single image
- process with default settings using the command: ./buoy_single_image.sh -d /waterline/
- follow the prompts to find the region of interest extents
- enter 1800 for top
- enter 1240 for left
- the waterline in mm down from the white reference mark on the scales is printed to screen
- view the detected waterline with: eog ./waterline/Test_Buoy_waterline.png
- view the other images created 
check errors with: cat ./waterline/Test_Buoy.err 

3. Process full video
- process video, specifying the ROI extents we found:
	./buoy_float_level.sh -d /data/hendreya/test_buoy_ARP/waterline/ -a 1800 -b 1240 	
	(use -r 2 to extract only 2 frames per second to speed up the processing)
- check the results in /waterline/output/ 
- results should match the example results in /example/output

