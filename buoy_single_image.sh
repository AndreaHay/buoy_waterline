#! /bin/bash

############################################################
#
# Script to test setting for buoy_float_level.sh
#
# A. Hay - 230130
#
############################################################



# DEFAULT VALUES - CHANGE THESE WITH COMMAND LINE FLAGS
# directory of video file
images_dir='/waterline_images/test_image/'
# location of scale coordinates
ref_coords='./scale_coordinates.csv'
#radius of cicles (in pixels)
circle_size_min=15
circle_size_max=40
colour='snow'
circle_sens=0.25
colour_fuzz=20
waterline_threshold_min=10
name='Test_Buoy' 
rate_fps=5
skip='f'
enhance='t'
contrast=1

while getopts d:m:x:r:c:s:f:t:n:z:e:o:h flag
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

echo " THIS IS TO TEST OUT SETTINGS, NOT TO PROCESS THE DATA (use buoy_float_level.sh for that)"
echo "everything other than the video file in the given dir will be deleted - continue (y/n)?"
read answer
if [ $answer == "n" ]; then
        echo "Exiting"
        exit 1
fi

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
echo -e "enhance images (-e) t/f: $enhance \n"

# delete everything except .MOV file in dir
find ${images_dir}* ! -name "*.MOV" -delete

# move to images dir
cd $images_dir

# create one image file from the video
echo "creating image from video"
ffmpeg -i $images_dir/*.MOV -vf "select=eq(n\,50)" -vframes 1 ${name}.png > $images_dir/ffmpeg.log 2> $images_dir/ffmpeg.err

file=${name}.png


# enhance the images
if [ $enhance == 't' ]; then
	echo "enhancing image, contrast = ${contrast}"
	cp $file ${file/.png/_orig.png}
	enhance_contrast.py $file $contrast >contrast.log 2>contrast.err
fi

# extract colour
echo "extracting colour, color = ${colour}, threshold = ${colour_fuzz}"
convert $file -channel rgba -fuzz ${colour_fuzz}% -fill none +opaque $colour ./${file/.png/_${colour}.png}

echo "running python script now"

buoy_single_image.py $images_dir $circle_size_min $circle_size_max $ref_coords $colour $circle_sens $name $waterline_threshold_min $enhance ${file/.png/_${colour}.png} ${file/.png/_${colour}} 2> ${name}.err
echo "done!"
echo "any errors from python:"
cat ${name}.err