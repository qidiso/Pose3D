#!/bin/bash

# Author: An Liang
# Date  : 2018 Jan 12 
# Usage: 
#     put this script in the folder of mpii_inf_3dhp dataset
#     and run it, to extract videos to images 
S=1
seqname=("Seq1" "Seq2")
while (( $S<=8 )) 
do 
    echo "processing subject $S" 
    for seq in ${seqname[*]}
    do 
        X=0
        folder="S${S}/${seq}/imageSequence"
        while (( $X<=8 ))
        do 
            echo $X
            videofile="${folder}/video_${X}.avi"
            imagefolder="${folder}/video_${X}"
            if [ -d "${imagefolder}" ];then 
                echo "folder ${imagefolder} has existed."  
                ffmpeg -i "${videofile}" -qscale:v 1 "${imagefolder}/img_${X}_%06d.jpg" 
            elif [ -f "${videofile}" ];then 
                mkdir "${imagefolder}"
                #ffmpeg -i "${videofile}" -qscale:v 1 "${imagefolder}/img_${X}_%06d.jpg" 
                echo "Y  $X" 
            else 
                echo "Not exist: " 
                echo "${videofile}"
            fi 

            let "X++"
        done
    done
    let "S++"
done 
