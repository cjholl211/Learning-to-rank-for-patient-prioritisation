# Learning to rank for patient prioritisation
MSc research project 

This is an application of a pairwise learning to rank algorithm named DirectRanker. A random forest classifier is also used to ensure that the data contains enough signal to predict patient severity.

## Dependencies
The following libraries are required: numpy, pandas, sklearn, matplotlib and tensorflow.

Please visit [Physionet](https://mimic.physionet.org/) for access to the MIMIC-III database. Run the SQL scripts found [here](https://github.com/MIT-LCP/mimic-code/tree/master/concepts/severityscores) to calculate severity scores. In turn, this will produce 'first day' data which is utilised throughout this study.
