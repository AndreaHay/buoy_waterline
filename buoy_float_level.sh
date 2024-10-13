#! /bin/bash

############################################################
#
# Workflow script to determine floating level of buoys
#
# A. Hay - 210607
#
############################################################

echo "starting"
echo -e "*** Don't forget to set ROI_top and ROI_left with -a and -b! (use buoy_single_image.sh to find) ***"

# location of scale coordinates
ref_coords='/scale_coordinates.csv'

# variables to be set by user using command line flags
# some default values below

images_dir=./ # directory of photos - use full path
circle_size_min=15 #radius of cicles (in pixels)
circle_size_max=40 #radius of cicles (in pixels)
colour='snow' # colour to extract
circle_sens=0.25 # circle detection threshold (lower means more circles will be detected)
colour_fuzz=20 # colour extraction threshold (lower means tighter tolerance (i.e. less area will be extracted))
waterline_threshold_min=10 # minimum waterline extraction threshold (lower means fainter waterline will be detected)
name='Test_Buoy' # e.g. 12A
rate_fps=5 # rate (frames per second to extract from video, can be non int value)
enhance='t' # enhance images (t/f) - if the difference between the waterline and the scale is not clear, try this
contrast=1 # if enhance is t, this is the contrast value to use (higher means more contrast)
ROI_top=1750

while getopts d:m:x:r:c:s:f:t:n:z:e:o:a:b:h flag
do
	case "${flag}" in
		d) images_dir=${OPTARG};;
		m) circle_size_min=${OPTARG};;
		x) circle_size_max=${OPTARG};;
		r) rate_fps=${OPTARG};;
		c) colour=${OPTARG};;
		s) circle_sens=${OPTARG};;
		f) colour_fuzz=${OPTARG};;
		t) waterline_threshold_min=${OPTARG};;
		n) name=${OPTARG};;
		z) skip=${OPTARG};;
		e) enhance=${OPTARG};;
		o) contrast=${OPTARG};;
		a) ROI_top=${OPTARG};;
		b) ROI_left=${OPTARG};;
		h) 
			echo -e "\n SETTINGS:"
			echo "images directory (-d) (USE FULL DIR PATH) : $images_dir"
			echo "min circle size in pixels (-m) : $circle_size_min"
			echo "max circle size in pixels (-x) : $circle_size_max"
			echo "colour to extract (-c) : $colour"
			echo "circle detection threshold (-s) : $circle_sens"
			echo "colour extraction threshold (-f) : $colour_fuzz"
			echo "name - no spaces please (-n) : $name"
			echo "rate (frames per second to extract) (-r): $rate_fps"
			echo "minimum waterline extraction threshold (-t): $waterline_threshold_min"
			echo "skip image extraction from video (-z) t/f: $skip"
			echo "contrast (-o) : $contrast"
			echo "ROI top (-a) : $ROI_top"
			echo "ROI left (-b) : $ROI_left"
			echo -e "enhance images (-e) t/f: $enhance \n"
			exit 0
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
			;;
		:)
	esac
done
 
echo -e "\n SETTINGS:"
echo "images directory (-d) (USE FULL DIR PATH) : $images_dir"
echo "min circle size in pixels (-m) : $circle_size_min"
echo "max circle size in pixels (-x) : $circle_size_max"
echo "colour to extract (-c) : $colour"
echo "circle detection threshold (-s) : $circle_sens"
echo "colour extraction threshold (-f) : $colour_fuzz"
echo "name - no spaces please (-n) : $name"
echo "rate (frames per second to extract) (-r): $rate_fps"
echo "minimum waterline extraction threshold (-t): $waterline_threshold_min"
echo "skip image extraction from video (-z) t/f: $skip"
echo "contrast (-o) : $contrast"
echo "ROI top (-a): $ROI_top"
echo "ROI left (-b): $ROI_left"
echo -e "enhance images (-e) t/f: $enhance \n"

#if images files don't exit, create them from the video
if [ ! "$(ls -A $images_dir/$name*.png 2>/dev/null)" ]; then
	echo "creating images from video"
	ffmpeg -i $images_dir/*.MOV -r $rate_fps $images_dir/$name%04d.png > $images_dir/ffmpeg.log 2> $images_dir/ffmpeg.err
else
  echo "using the following images:"
  ls $images_dir/$name*.png
fi

# extract coloured areas from all .png files in the images directory
cd $images_dir
mkdir -p output
cd output
mkdir -p ${colour}
mkdir -p edges
mkdir -p circle_detection
mkdir -p rectified_images
mkdir -p ROI
mkdir -p waterline
mkdir -p failed

echo "image_no,waterline" > ${name}_results.csv
echo "ID,mean,median,stdev,std_err,Q1,Q3,IQR,no_waterlines_detected,no_images_total" > ${name}_final_stats.csv

cd -

# if we want to increase the contrast
if [ $enhance == 't' ]; then
	echo "enhancing images"
	mkdir original_images
	for file in *.png; do
	  cp $file ./original_images/${file/.png/_orig.png}
		enhance_contrast.py $file $contrast >contrast.log 2>contrast.err
	done
fi


# if colour hasn't been extracted, do it
if [ ! "$(ls -A "$images_dir/output/${colour}"/"${name}"*.png 2>/dev/null)" ]; then
	echo "extracting colour"
	for FILE in *.png; do
		convert $FILE -channel rgba -fuzz ${colour_fuzz}% -fill none +opaque $colour ./output/${colour}/${FILE/.png/_${colour}.png}
	done
fi

cd -

buoy.py $images_dir $circle_size_min $circle_size_max $ref_coords $colour $circle_sens $name $waterline_threshold_min $enhance $ROI_top $ROI_left > ${name}.log 2> ${name}.err


cd ./

sed "2 s/./${name},&/" ${name}_final_stats.csv > temp.csv
mv temp.csv ${name}_final_stats.csv

# print error file
cat ${name}.err
