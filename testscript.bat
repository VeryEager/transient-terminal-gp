time /T
cd code/
:: Testing of each dataset, on each method, in succession
python test-script.py winequality-red quality ; sgp
python test-script.py winequality-red quality ; mogp
python test-script.py winequality-red quality ; ttgp
python test-script.py winequality-white quality ; sgp
python test-script.py winequality-white quality ; mogp
python test-script.py winequality-white quality ; ttgp
python test-script.py houseprice-boston MEDV , sgp
python test-script.py houseprice-boston MEDV , mogp
python test-script.py houseprice-boston MEDV , ttgp
python test-script.py strength-concrete csMPa , sgp
python test-script.py strength-concrete csMPa , mogp
python test-script.py strength-concrete csMPa , ttgp
python test-script.py estcount-bike count , sgp
python test-script.py estcount-bike count , mogp
python test-script.py estcount-bike count , ttgp
time /T