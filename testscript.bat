time /T
cd code/
echo ############## BEGIN winequality-red DATA ##############
python test-script.py winequality-red quality ; sgp
python test-script.py winequality-red quality ; mogp
python test-script.py winequality-red quality ; ttgp
echo ############## END winequality-red DATA ##############
time /T