#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# Mail me at the b(eginning) and e(nd) of the job
#
#$ -M stoutashe@myvuw.ac.nz
#$ -m be
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific
# directory that should have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
if [ -d /local/tmp/stoutashe/$JOB_ID ]; then
        cd /local/tmp/stoutashe/$JOB_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP STOUTASHE "
        ls -la /local/tmp/stoutashe
        echo "Exiting"
        exit 1
fi
#
# Now we are in the job-specific directory so now can do something useful
#
# Stdout from programs and shell echos will go into the file
#    scriptname.o$JOB_ID
#  so we'll put a few things in there to help us see what went on
#
echo ==UNAME==
uname -n
echo ==WHO AM I and GROUPS==
id
groups
echo ==SGE_O_WORKDIR==
echo $SGE_O_WORKDIR
echo ==/LOCAL/TMP==
ls -ltr /local/tmp/
#
# OK, where are we starting from and what's the environment we're in
#
echo ==RUN HOME==
pwd
ls
echo ==ENV==
env
echo ==SET==
set
#
echo == WHATS IN LOCAL/TMP ON THE MACHINE WE ARE RUNNING ON ==
ls -ltra /local/tmp | tail
#
echo == WHATS IN LOCAL TMP STOUTASHE JOB_ID AT THE START==
ls -la 
#
# Copy the required files to the local directory
#
cp -r /am/courtenay/home1/stoutashe/Desktop/transient-terminal-gp .
cd transient-terminal-gp/code
echo ==WHATS THERE HAVING COPIED STUFF OVER AS INPUT==
ls -la 
# 
# Run the test script using Python
#
python test-ttsthresh.py winequality-red quality \; 
#python test-ttsthresh.py winequality-white quality \; 
#python test-ttsthresh.py strength-concrete csMPa \, 
#python test-ttsthresh.py houseprice-boston MEDV \, 

#
echo ==AND NOW, HAVING DONE SOMTHING USEFUL AND CREATED SOME OUTPUT==
ls 
#
# Save the data results into a folder for later use
#
cd ..
cd docs/Data
ls
cd ..
cd ..
cp -r docs/Data /am/courtenay/home1/stoutashe/Desktop/Results

#
echo "Ran through OK"

