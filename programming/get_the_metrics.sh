#!/bin/bash

# cd predictions
# copy the prediction file and the ground truth file from the current directory to the directory where the metrics are calculated
# the prediction file is the first argument and the ground truth file is the second argument
prediction_file=$1
ground_truth_file=$2
name=$3
number=$4
with_sps_metric=$5

# path to the directory where the metrics are calculated
path_to_metrics="${HOME}/Documents/Mathematik/24 FS/Semester_Paper_DCML/CHORD_EVAL/"
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
prediction_file_name="prediction.pkl"
ground_truth_file_name="ground_truth.pkl"

prediction_file_dest="$path_to_metrics$prediction_file_name"
ground_truth_file_dest="$path_to_metrics$ground_truth_file_name"


# get the parent path of the current directory

# move the files
echo "copying files to"
echo $path_to_metrics
cp -f "$prediction_file" "$prediction_file_dest"
cp -f "$ground_truth_file" "$ground_truth_file_dest"

# run the metric_main.py script with python 3.8.5 with pyenv
cd "$path_to_metrics"
pyenv local chord_eval
echo 'running metric_main.py with python 3.8.5'
~/.pyenv/versions/chord_eval/bin/python "${path_to_metrics}metric_main.py" $number $with_sps_metric
echo 'scores calculated and saved'

# copy the scores back to the original directory
# the scores are saved as 'scores_' + distance_name + '.npy', whith variable distance_name
# write it, so the actual distance name doesn't matter
# e.g. copy all files that start with 'scores_' and end with '.npy'
cp -f "${path_to_metrics}scores_"*.npy "$parent_path""/scores"
cd "$parent_path""/scores"


# list_of_files=$(find . \(-name "*_${number}.npy" -not -name "*${name}*"\))
# echo $list_of_files
# # for f in *_$number.npy;
# for f in $list_of_files;
#   # do printf '%s\n' "${f%.npy}_${name}_${number}.npy";
#   do mv -f "$f" "${f%.npy}_num_${number}_${name}.npy";
# done

echo 'scores copied back to the scores directory'

