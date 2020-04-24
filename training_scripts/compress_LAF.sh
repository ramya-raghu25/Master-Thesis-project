"""
DESCRIPTION:     Shell script to compress the Lost and Found dataset from original 2048x1024 PNG to 1024x512 WEBP
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

# Please download a webp encoder from https://anaconda.org/conda-forge/libwebp
# Please download imagemagick from https://imagemagick.org/index.php

# $LAF_dir = directory of original dataset
# $LAF_compressed_dir = output directory

LAF_dir=Z:/Master_Thesis_Ramya/datasets/LAF
LAF_compressed_dir=Z:/Master_Thesis_Ramya/datasets/LAF

python compress_images.py \
	$LAF_dir/leftImg8bit \
	$LAF_compressed_dir/small_leftImg8bit \
	"cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6 -resize 1024 512" \
	--ext ".webp" --concurrent 20

python compress_images.py \
	$LAF_dir/gtCoarse \
	$LAF_compressed_dir/small_gtCoarse \
	"magick {src} -filter point -resize 50% {dest}" \
	--ext ".png" --concurrent 20
