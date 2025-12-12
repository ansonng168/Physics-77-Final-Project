1. simulate.py:
   This script generates toy Higgs-like invariant mass datasets for different integrated luminosities. To use it, run the script, enter a comma-separated list of luminosities (e.g. 0.1,1,5), and optionally adjust the signal mass, width, amplitude, and random seed; the code will output one CSV file per luminosity containing expected and observed background, signal, and total counts in each mass bin. Simulated datasets used in our analysis has already been generated with the default values, only changing the luminosity.

2. simulationvalidation.py:

3. frequentist.py:
   This script performs a frequentist hypothesis test on the simulated mass spectra produced by the generator. To use it, run the script, provide the path to a CSV dataset, and the code will print the local and global p-values and significances for that dataset.

4. bayesian.py:
   
   

