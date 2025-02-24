# ECON148 Project 03
This is the repository for the third project in the course ECON 148 Data Science for Economists Spring 2024 at the University of California, Berkeley.

This assignment was to reproduce the analysis of a paper by Edward Miguel by translating the original paper's code from Stata to Python. The paper is "Sell Low and Buy High: Arbitrage and Local Price Effects in Kenyan Markets" by Marshall Burke, Lauren Falcao Bergquist, and Edward Miguel. The original paper can be found [here](https://economics.harvard.edu/files/economics/files/ms29141.pdf).

The data used in the project can be found on the Harvard Dataverse [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/C8UMQP). We are using the 1.2 version of the data published on Apr 09, 2019. The necessary datasets are also included in the data folder of this repository. However, the only difference between this and the 1.0 version published on Dec 23, 2018, is the addition of citation metadata.

The Python code is written and run through a Python 3.11.8 distribution and requires the following packages (version number I have used in parenthesis)

- ```pandas (2.1.4)```
- ```numpy (1.26.4)```
- ```scipy (1.11.4)```
- ```statsmodels (0.14.0)```
- ```matplotlib (3.8.3)```
- ```stargazer (0.0.7)```

Be aware that if running a pandas version older than 2.1.0, the `.map` method might not work as intended and will need to be replaced with the now deprecated `.applymap`. This is the only known issue when running the code on older versions.

Other than the README.md, the repository contains the following folders and files:
- [proj03.ipynb](proj03.ipynb): The main notebook containing the code for the project
- [proj03.py](proj03.py): The Python script for our wild cluster bootstrap-t implementation
- [ECON148_Project03.pdf](ECON148_Project03.pdf): The final report for the project
- [tables](tables): Contains the output tables as `.tex` generated by the code; a total of 8 tables are generated
- [figures](figures): Contains the output figures generated by the code; only 2 figures are generated
- [data](data): Contains all the datasets provided through the Harvard Dataverse. However, the only datasets used in the code are [baseline.dta](data/baseline.dta), [cleanpricedata_y1y2.dta](data/cleanPriceData_Y1Y2.dta), and [ms1ms2_pooled.dta](data/MS1MS2_pooled.dta)
- [Sell Low and Buy High: Arbitrage and Local Price Effects in Kenyan Markets.pdf](Sell%20Low%20and%20Buy%20High%3A%20Arbitrage%20and%20Local%20Price%20Effects%20in%20Kenyan%20Markets.pdf): The original paper by Marshall Burke, Lauren Falcao Bergquist, Edward Miguel
