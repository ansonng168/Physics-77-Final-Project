1. simulate.py:
   This script generates toy Higgs-like invariant mass datasets for different integrated luminosities. To use it, run the script, enter a comma-separated list of luminosities (e.g. 0.1,1,5), and optionally adjust the signal mass, width, amplitude, and random seed; the code will output one CSV file per luminosity containing expected and observed background, signal, and total counts in each mass bin. Simulated datasets used in our analysis has already been generated with the default values, only changing the luminosity.

2. simulationvalidation.ipynb:
   This python notebook contains the code for the validation of our toy experiment data. Datasets from the CERN open data portal are required to be downloaded to properly run the code. They can be accessed here: https://opendata.cern.ch/record/93915. Due to the large nature of these datasets, download time will be lengthy and take up a significant amount of storage. 

4. frequentist.py:
   This script performs a frequentist hypothesis test on the simulated mass spectra produced by the generator. To use it, run the script, provide the path to a CSV dataset, and the code will print the local and global p-values and significances for that dataset.

5. bayesian.py:
   This script performs a Bayesian analysis of mass spectrum data, comparing background-only and signal-plus-background models using Dynesty for nested sampling. To use it, run the script, provide a CSV file with observed mass spectrum data, and it will output the results along with plots of the fit and posterior distributions for both models.

6. comparision.py:
   This script compares the frequentist results with the Bayesian results. It is used in datacomparision.py.

7. datacomparision.py:
   This script batch-processes all simulated datasets (dataset_L*.csv) and summarizes the results of the frequentist and Bayesian analyses. For each luminosity, it calls the compare function to extract the local and global significances, Bayesian evidences, and the posterior probability of the background-only hypothesis. The results are collected into a single table and saved as summary_results.csv, allowing easy comparison of statistical sensitivity as a function of integrated luminosity.
   





