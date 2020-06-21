#!/bin/bash
#$ -N Sub_PrepareData
#$ -S /bin/bash
#$ -q iui.q
#$ -pe smp 1
#$ -cwd
#$ -o Sub_PrepareData.out
#$ -e Sub_PrepareData.err
#$ -M kyesilbek@ku.edu.tr
#$ -m ea

/share/apps/matlab/R2014b/bin/matlab -nodesktop -nosplash -nodisplay -r prepareData > Sub_PrepareData.txt
