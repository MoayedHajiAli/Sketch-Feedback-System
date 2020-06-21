#!/bin/bash
#$ -N Zernike
#$ -S /bin/bash
#$ -q iui.q
#$ -pe smp 1
#$ -cwd
#$ -o Zernike.out
#$ -e Zernike.err
#$ -M kyesilbek@ku.edu.tr
#$ -m ea

/share/apps/matlab/R2014b/bin/matlab -nodesktop -nosplash -nodisplay -r extractZernikefeats_main > Zernike.txt
