# SVM Implementation
Hand-written characters using support vector machine (SVM).

# Requirements
Python 3.7.6 is the version we used for training. Other packages include:
- matplotlib v3.1.3
- numpy v1.18.1
- scikit-learn v0.22.0
- pathlib v2.3.5
- pandas v1.0.3

# Pakcage Installation with Conda
Use 
```
conda install <package-name>=<package-version> 
```
Example:
```
conda install scikit-learn=0.22.0 
```

# Usage
In the current folder, run
```
python svm.py
```
A few lines that indicates the stages of the process are printed. After the learning is complete, a `classification_report.csv` and `confusion_matrix.csv` will be output.