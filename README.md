# Post-GWAS

This repository contains both source data, code, as well as results.
Please note that Pathway Studio is a commercial product, so neither the edgelist nor the neo4j database have been uploaded here.

Also, none of the packages used to create the feature sets (e.g. EVOKE, node2vec, rdf2vec, etc.) have been uploaded here.
Please refer to their respective GitHub repositories.

Each directory contains a method, within which both code as well as input and output files are contained.
Most of the methods use some version of "...classification.py" or an ipython notebook to perform the cross-validation.
The base directory contains code for comparing the classifications, as well as creating the ensemble method.

All the references to files in the code refer to the input files upon which the results were based.
One should therefore simply be able to run the code immediately to recreate the results shown in the manuscript.
