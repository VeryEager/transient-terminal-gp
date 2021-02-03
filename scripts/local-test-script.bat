time /T
cd code/
:: Testing of each dataset, on each method, in succession
python test-script.py winequality-red quality ; sogp 50 500
python test-script.py winequality-white quality ; sogp 50 500
python test-script.py strength-concrete csMPa , sogp 50 500
python test-script.py houseprice-boston MEDV , sogp 50 500

python test-script.py winequality-red quality ; sottgp 50 500
python test-script.py winequality-white quality ; sottgp 50 500
python test-script.py strength-concrete csMPa , sottgp 50 500
python test-script.py houseprice-boston MEDV , sottgp 50 500

python test-script.py winequality-red quality ; mogp 50 500
python test-script.py winequality-white quality ; mogp 50 500
python test-script.py strength-concrete csMPa , mogp 50 500
python test-script.py houseprice-boston MEDV , mogp 50 500

python test-script.py winequality-red quality ; ttgp 50 500
python test-script.py winequality-white quality ; ttgp 50 500
python test-script.py strength-concrete csMPa , ttgp 50 500
python test-script.py houseprice-boston MEDV , ttgp 50 500

python test-script.py winequality-red quality ; mogp 250 200
python test-script.py winequality-white quality ; mogp 250 200
python test-script.py strength-concrete csMPa , mogp 250 200
python test-script.py houseprice-boston MEDV , mogp 250 200

python test-script.py winequality-red quality ; ttgp 250 200
python test-script.py winequality-white quality ; ttgp 250 200
python test-script.py strength-concrete csMPa , ttgp 250 200
python test-script.py houseprice-boston MEDV , ttgp 250 200

time /T