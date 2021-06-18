#!/usr/bin/env python
# coding: utf-8

# Load the required packages
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import json
import numpy as np
from itertools import product

ML_algorithms = ["LR", "SVM", "DT", "KNN", "RF"]
variations = ["normal", "graphlet", "autoencode", "combi"]
all_bp_distances = [100, 500, 1000, 2000]
refsets = ["farashi", "farashi p-value cutoff", "DeRycke", "Teslovich"]
weighted = ["weighted", "unweighted"]

all_metrics = pd.DataFrame(list(product(refsets, ML_algorithms, variations, weighted, all_bp_distances)), columns = ["refset", "algorithm", "variation", "weighted", "bp distance"])

for am_index, am_values in all_metrics.iterrows():
    print("Predicting row " + str(am_index) + " of " + str(len(all_metrics)))

    classifier = am_values["algorithm"]
    class_type = am_values["variation"]

    if am_values["weighted"] == "weighted":

        # Load the data
        if class_type == "normal":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_weighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)

        # Load the autoencoded embeddings
        if class_type == "autoencode":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/autorcode_weighted_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_weighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            f.index = f2.index


        # Extend embeddings with graphlets
        if class_type == "graphlet":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_weighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            f = f.merge(graphlets, right_index = True, left_index = True)

        if class_type == "combi":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/autorcode_weighted_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_weighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            f.index = f2.index
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            f = f.merge(graphlets, right_index = True, left_index = True)

    if am_values["weighted"] == "unweighted":

        # Load the data
        if class_type == "normal":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_unweighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)

        # Load the autoencoded embeddings
        if class_type == "autoencode":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/autorcode_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_unweighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            f.index = f2.index


        # Extend embeddings with graphlets
        if class_type == "graphlet":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_unweighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            f = f.merge(graphlets, right_index = True, left_index = True)

        if class_type == "combi":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/autorcode_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/PathwayStudio_PPI_complete_directed_unweighted.emb", sep = " ", skiprows = 1, header = None, index_col = 0)
            f.index = f2.index
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            f = f.merge(graphlets, right_index = True, left_index = True)



    # Load the mapping file
    with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
        ensembl_dict = json.load(fp)

    # Load the reference set
    if am_values["refset"] == "farashi":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Farashi/Farashi full 2000000 bp distance no pvalue filtering.csv")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["gene_ids"]]

    if am_values["refset"] == "farashi p-value cutoff":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Farashi/Farashi full 2000000 bp distance no pvalue filtering.csv")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["gene_ids"]]
        ref = ref[ref["GWAS/eQTL p-value¥"] <= float("5e-8")]

    if am_values["refset"] == "DeRycke":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/DeRycke/DeRycke reference set.csv")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["ENSEMBL"]]


    if am_values["refset"] == "Teslovich":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Teslovich/Teslovich reference set.csv")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["ENSEMBL"]]


    # Drop all unmappable candidates
    ref.dropna(subset = ["nodeID"], inplace = True)
    ref["nodeID"] = ref["nodeID"].astype(int)


    # In[9]:


    # Set bp distance cutoff
    max_bp_distance = am_values["bp distance"]
    max_bp_distance = max_bp_distance * 1000
    ref = ref[ref["bp distance absolute"] <= max_bp_distance]


    # In[10]:


    # Drop all SNPs which no longer have a positive case
    pos_counts = ref.groupby("SNP ID")["Class"].sum()
    ref = ref[~ref["SNP ID"].isin(pos_counts[pos_counts == 0].index)]


    # In[11]:


    # Identify all genes which are at least once positive
    positives = ref.groupby("nodeID")["Class"].sum()
    positives[positives > 1] = 1

    f = f.merge(positives, left_index = True, right_index = True)


    # ## Leave SNP out classification

    # In[12]:


    outcomes = pd.DataFrame()
    train_auc_score = []
    train_auc_rank = []


    # In[13]:


    # Perform leave-SNP-out cross validation
    SNPs = list(set(ref["SNP ID"]))
    for snp in SNPs:
        print("Predicting candidates for " + snp + ", number " + str(SNPs.index(snp) + 1) + " out of " + str(len(SNPs)))

        f_test = f[f.index.isin(ref[ref["SNP ID"] == snp]["nodeID"])].copy()
        f_train = f[f.index.isin(ref[ref["SNP ID"] != snp]["nodeID"])].copy()

        train_class = f["Class"][f.index.isin(f_train.index)]
        test_class = f["Class"][f.index.isin(f_test.index)]

        f_test.drop(columns = ["Class"], inplace = True)
        f_train.drop(columns = ["Class"], inplace = True)

        if classifier == "SVM":
            clf = SVR(gamma="auto")
        if classifier == "DT":
            clf = DecisionTreeRegressor()
        if classifier == "KNN":
            clf = KNeighborsClassifier(n_neighbors = 3)
        if classifier == "LR":
            from warnings import filterwarnings
            filterwarnings('ignore')
            clf = LogisticRegression()
        if classifier == "RF":
            clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, max_features = "sqrt", max_depth = 5)


        clf.fit(f_train, train_class)

        outcomes = pd.concat([outcomes, pd.DataFrame({"predicted" : clf.predict(f_test),
                                                        "SNP ID" : snp,
                                                        "nodeID" : f_test.index})])

    outcomes = outcomes.merge(ref[["SNP ID", "nodeID", "Class"]], on = ["SNP ID", "nodeID"], how = "left")


    # ## Leave chromosome out validation

    # In[14]:


    outcomes2 = pd.DataFrame()
    train_auc_score2 = []
    train_auc_rank2 = []


    # In[15]:


    # Perform leave-SNP-out cross validation
    chromosomes = list(set(ref["chromosome"]))

    for chrom in chromosomes:
        print("Predicting candidates for chromosome " + str(chrom))

        f_test = f[f.index.isin(ref["nodeID"][ref["chromosome"] == chrom])].copy()
        f_train = f[f.index.isin(ref["nodeID"][ref["chromosome"] != chrom])].copy()

        train_class = f["Class"][f.index.isin(f_train.index)]
        test_class = f["Class"][f.index.isin(f_test.index)]

        f_test.drop(columns = ["Class"], inplace = True)
        f_train.drop(columns = ["Class"], inplace = True)

        if classifier == "SVM":
            clf = SVR(gamma="auto")
        if classifier == "DT":
            clf = DecisionTreeRegressor()
        if classifier == "KNN":
            clf = KNeighborsClassifier(n_neighbors = 3)
        if classifier == "LR":
            from warnings import filterwarnings
            filterwarnings('ignore')
            clf = LogisticRegression()
        if classifier == "RF":
            clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, max_features = "sqrt", max_depth = 5)

        clf.fit(np.array(f_train), np.array(train_class))

        outcomes2 = pd.concat([outcomes2, pd.DataFrame({"predicted" : clf.predict(f_test),
                                                        "Class" : test_class,
                                                        "chromosome" : chrom,
                                                        "nodeID" : f_test.index})])


    # In[16]:


    #outcomes.to_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/Leave-SNP-Out cross validation weighted " + class_type + " " + str(max_bp_distance) + " " + classifier + ".csv", index = False)
    #outcomes2.to_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/Leave-chromosome-Out cross validation weighted " + class_type + " " + str(max_bp_distance) + " " + classifier + ".csv", index = False)


    # ## Evaluate leave-SNP-out

    # In[17]:


    outcomes = outcomes.sort_values(["SNP ID", "predicted"], ascending = False)
    outcomes["For-SNP rank"] = outcomes.groupby("SNP ID").cumcount() + 1


    # In[18]:


    len(set(outcomes["SNP ID"]))


    # In[19]:


    sum(outcomes["Class"])


    # In[20]:


    import sklearn.metrics

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes["Class"], -outcomes["For-SNP rank"], pos_label = 1)
    all_metrics.at[am_index, "ROC-AUC overall (lso)"] = sklearn.metrics.auc(fpr, tpr) * 100


    # In[21]:


    # Calculate the ROC-AUC for every SNP and average the result
    SNPS2 = list(set(outcomes["SNP ID"]))
    aucs = []
    for snp in SNPS2:
      if len(set(outcomes["Class"][outcomes["SNP ID"] == snp])) == 1:
          aucs.append(list(set(outcomes["Class"][outcomes["SNP ID"] == snp]))[0])
      else:
          fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes["Class"][outcomes["SNP ID"] == snp], -outcomes["For-SNP rank"][outcomes["SNP ID"] == snp], pos_label = 1)
          aucs.append(sklearn.metrics.auc(fpr, tpr))
    all_metrics.at[am_index, "ROC-AUC - mean per snpl (lso)"] = sum(aucs)/len(aucs)


    # In[22]:


    # Calculate hits @1
    all_metrics.at[am_index, "Hits@1(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] == 1)])


    # In[23]:


    # Calculate hits @3
    all_metrics.at[am_index, "Hits@3(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 3)])


    # In[24]:


    # Calculate hits @5
    all_metrics.at[am_index, "Hits@5(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 5)])


    # In[25]:


    # Calculate hits @10
    all_metrics.at[am_index, "Hits@10(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 10)])


    # In[26]:


    all_metrics.at[am_index, "Mean rank (lso)"] = outcomes["For-SNP rank"][(outcomes["Class"] == 1)].mean()


    # In[27]:


    all_metrics.at[am_index, "Median rank (lso)"] = outcomes["For-SNP rank"][outcomes["Class"] == 1].quantile(q = [0,0.25,0.5,0.75,1])[0.50]


    # ## Evaluate leave-chromosome-out

    # In[28]:


    outcomes2 = outcomes2.sort_values(["chromosome", "predicted"], ascending = False)
    outcomes2["For-chromosome rank"] = outcomes2.groupby("chromosome").cumcount() + 1


    # In[29]:


    chromosomes = list(set(outcomes2["chromosome"]))
    aucs = []
    for chrom in chromosomes:
      fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes2["Class"][outcomes2["chromosome"] == chrom], -outcomes2["For-chromosome rank"][outcomes2["chromosome"] == chrom], pos_label = 1)
      aucs.append(sklearn.metrics.auc(fpr, tpr))
    #print(sum(aucs)/len(aucs))


    # In[30]:


    ref = ref.merge(outcomes2[["nodeID", "predicted"]], on = "nodeID", how = "left")


    # In[31]:


    ref = ref.sort_values(["SNP ID", "predicted"], ascending = False)
    ref["For-SNP rank"] = ref.groupby("SNP ID").cumcount() + 1


    # In[32]:


    fpr, tpr, thresholds = sklearn.metrics.roc_curve(ref["Class"], -ref["For-SNP rank"], pos_label = 1)
    all_metrics.at[am_index, "ROC-AUC overall (lco)"] = sklearn.metrics.auc(fpr, tpr) * 100


    # In[33]:


    # Calculate the ROC-AUC for every SNP and average the result
    SNPS2 = list(set(ref["SNP ID"]))
    aucs = []
    for snp in SNPS2:
      if len(set(ref["Class"][ref["SNP ID"] == snp])) == 1:
          aucs.append(list(set(ref["Class"][ref["SNP ID"] == snp]))[0])
      else:
          fpr, tpr, thresholds = sklearn.metrics.roc_curve(ref["Class"][ref["SNP ID"] == snp], -ref["For-SNP rank"][ref["SNP ID"] == snp], pos_label = 1)
          aucs.append(sklearn.metrics.auc(fpr, tpr))
    all_metrics.at[am_index, "ROC-AUC - mean per snpl (lco)"] = sum(aucs)/len(aucs)


    # In[34]:


    # Calculate hits @1
    all_metrics.at[am_index, "Hits@1(lco)"] = sum(ref["Class"][(ref["Class"] == 1) & (ref["For-SNP rank"] == 1)])


    # In[35]:


    # Calculate hits @3
    all_metrics.at[am_index, "Hits@3(lco)"] = sum(ref["Class"][(ref["Class"] == 1) & (ref["For-SNP rank"] <= 3)])


    # In[36]:


    # Calculate hits @5
    all_metrics.at[am_index, "Hits@5(lco)"] = sum(ref["Class"][(ref["Class"] == 1) & (ref["For-SNP rank"] <= 5)])


    # In[37]:


    # Calculate hits @10
    all_metrics.at[am_index, "Hits@10(lco)"] = sum(ref["Class"][(ref["Class"] == 1) & (ref["For-SNP rank"] <= 10)])


    # In[38]:


    all_metrics.at[am_index, "Mean rank (lco)"] = ref["For-SNP rank"][(ref["Class"] == 1)].mean()


    # In[39]:


    all_metrics.at[am_index, "Median rank (lco)"] = ref["For-SNP rank"][ref["Class"] == 1].quantile(q = [0,0.25,0.5,0.75,1])[.50]


all_metrics.to_csv("/Users/vlietstraw/git/Post-GWAS/Struc2vec/all_variations_performance_metrics.csv", sep = ";", decimal = ",", index = False)
