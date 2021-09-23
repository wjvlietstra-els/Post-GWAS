#!/usr/bin/env python
# coding: utf-8

# # Node2vec classification

# In[1]:


# Load the required packages
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import json
import numpy as np
from itertools import product
from datetime import datetime
from sklearn.utils import shuffle
import sklearn.metrics

writeOut = True

ML_algorithms = ["LR", "SVM", "DT", "KNN1", "KNN3", "KNN5", "KNN7", "KNN9", "RF"]
variations = ["normal", "log"]
all_bp_distances = [25, 50, 100, 500, 1000, 2000, "depict"]
refsets = ["Teslovich", "DeRycke", "farashi", "farashi p-value cutoff"]

all_metrics = pd.DataFrame(list(product(refsets, ML_algorithms, variations, all_bp_distances)), columns = ["refset", "algorithm", "variation", "bp distance"])

if writeOut:
    all_metrics = all_metrics[all_metrics["bp distance"] == "depict"]
    all_metrics = all_metrics[((all_metrics["variation"].isin(["normal"])) & (all_metrics["algorithm"] == "RF") & (all_metrics["refset"] == "farashi")) |
                ((all_metrics["variation"].isin(["log"])) & (all_metrics["algorithm"] == "KNN7") & (all_metrics["refset"] == "farashi p-value cutoff")) |
                ((all_metrics["variation"].isin(["log"])) & (all_metrics["algorithm"] == "DT") & (all_metrics["refset"] == "DeRycke")) |
                ((all_metrics["variation"].isin(["log"])) & (all_metrics["algorithm"] == "LR") & (all_metrics["refset"] == "Teslovich"))
               ]

for am_index, am_values in all_metrics.iterrows():
    print("Predicting row " + str(am_index) + " of " + str(len(all_metrics)))
    # In[2]:

    classifier = am_values["algorithm"]
    class_type = am_values["variation"]

    # Run the graphlets
    f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
    f.drop(columns = [73], inplace = True)
    f.columns = ["Graphlet " + str(x) for x in range(len(list(f)))]
    f.index = [x + 1 for x in list(f.index)]

    if class_type == "log":
        f = f.apply(lambda x: np.log10(x, where = x > 0))

    # In[7]:

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
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/DeRycke/DeRycke reference set.csv", delimiter = ";")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["ENSEMBL"]]


    if am_values["refset"] == "Teslovich":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Teslovich/Teslovich reference set.csv")
        ref["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in ref["ENSEMBL"]]

    # In[8]:

    ref = shuffle(ref)

    # Set bp distance cutoff
    if am_values["bp distance"] != "depict":
        max_bp_distance = am_values["bp distance"]
        max_bp_distance = max_bp_distance * 1000
        ref = ref[ref["bp distance absolute"] <= max_bp_distance]
    elif am_values["bp distance"] == "depict":
        if am_values["refset"] == "farashi":
            depict = pd.read_csv("~/git/DEPICT/outcomes/Farashi complete 2nd round/farashi_no_pvalue_filtering_geneprioritization.txt", sep = "\t")
        if am_values["refset"] == "farashi p-value cutoff":
            depict = pd.read_csv("~/git/DEPICT/outcomes/Farashi complete 2nd round/farashi_default_pvalue_filtering_geneprioritization.txt", sep = "\t")
        if am_values["refset"] == "DeRycke":
            depict = pd.read_csv("~/git/DEPICT/outcomes/DeRycke/DeRycke_output_geneprioritization.txt", sep = "\t")
        if am_values["refset"] == "Teslovich":
            depict = pd.read_csv("~/git/DEPICT/outcomes/Teslovich for paper Wytze/Teslovich_output_geneprioritization.txt", sep = "\t")
        depict["nodeID"] = [ensembl_dict[x] if x in ensembl_dict.keys() else None for x in depict["Ensembl gene ID"]]


        depict["Locus"] = depict["Locus"].astype(str).apply(lambda x: x.split(";"))
        depict = depict.explode("Locus")

        snp_replacement_dict = {"rs113645266" : "rs6557271",
                        "rs150282463" : "rs13137700",
                        "rs67276543" : "rs34884832"}
        depict["Locus"] = depict["Locus"].replace(snp_replacement_dict)

        depict = depict[["Locus", "nodeID"]]
        depict.columns = ["SNP ID", "nodeID"]

        ref = ref.merge(depict, on = ["SNP ID", "nodeID"], how = "inner")

    # Drop all unmappable candidates
    ref.dropna(subset = ["nodeID"], inplace = True)
    ref["nodeID"] = ref["nodeID"].astype(int)


    # In[10]:


    # Drop all SNPs which no longer have a positive case
    pos_counts = ref.groupby("SNP ID")["Class"].sum()
    ref = ref[~ref["SNP ID"].isin(pos_counts[pos_counts == 0].index)]


    # In[11]:


    # ## Leave SNP out classification

    # In[12]:


    # outcomes = pd.DataFrame()
    # train_auc_score = []
    # train_auc_rank = []
    #
    #
    # # In[13]:
    #
    #
    # # Perform leave-SNP-out cross validation
    # SNPs = list(set(ref["SNP ID"]))
    # for snp in SNPs:
    #     print("Predicting candidates for " + snp + ", number " + str(SNPs.index(snp) + 1) + " out of " + str(len(SNPs)))
    #
    #     # Identify all genes which are at least once positive
    #     positives = ref[ref["SNP ID"] != snp].groupby("nodeID")["Class"].sum()
    #     positives[positives > 1] = 1
    #
    #     f_test = f[f.index.isin(ref[ref["SNP ID"] == snp]["nodeID"])].copy()
    #     f_train = f[f.index.isin(ref[ref["SNP ID"] != snp]["nodeID"])].copy()
    #
    #     f_train = f_train.merge(positives, left_index = True, right_index = True)
    #     f_test = f_test.merge(ref[["nodeID", "Class"]], how = "left", left_index = True, right_on = "nodeID")
    #
    #     train_class = f_train["Class"]
    #     test_class = f_test["Class"]
    #
    #     f_test.drop(columns = ["Class", 'nodeID'], inplace = True)
    #     f_train.drop(columns = ["Class"], inplace = True)
    #
    #     if classifier == "SVM":
    #         clf = SVR(gamma="auto")
    #     if classifier == "DT":
    #         clf = DecisionTreeRegressor()
    #     if classifier == "KNN1":
    #         clf = KNeighborsRegressor(n_neighbors = 1)
    #     if classifier == "KNN3":
    #         clf = KNeighborsRegressor(n_neighbors = 3)
    #     if classifier == "KNN5":
    #         clf = KNeighborsRegressor(n_neighbors = 5)
    #     if classifier == "KNN7":
    #         clf = KNeighborsRegressor(n_neighbors = 7)
    #     if classifier == "KNN9":
    #         clf = KNeighborsRegressor(n_neighbors = 9)
    #     if classifier == "LR":
    #         from warnings import filterwarnings
    #         filterwarnings('ignore')
    #         clf = LogisticRegression()
    #     if classifier == "RF":
    #         clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, max_features = "sqrt", max_depth = 5)
    #
    #     clf.fit(f_train, train_class)
    #
    #     outcomes = pd.concat([outcomes, pd.DataFrame({"predicted" : clf.predict(f_test),
    #                                                     "SNP ID" : snp,
    #                                                     "nodeID" : f_test.index})])
    #
    # outcomes = outcomes.merge(ref[["SNP ID", "nodeID", "Class"]], on = ["SNP ID", "nodeID"], how = "left")


    # ## Leave chromosome out validation

    # In[14]:


    outcomes2 = pd.DataFrame()
    train_auc_score2 = []
    train_auc_rank2 = []


    # In[15]:

    # Identify all genes which are at least once positive
    positives = ref.groupby("nodeID")["Class"].sum()
    positives[positives > 1] = 1

    f = f.merge(positives, left_index = True, right_index = True)


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
        if classifier == "KNN1":
            clf = KNeighborsRegressor(n_neighbors = 1)
        if classifier == "KNN3":
            clf = KNeighborsRegressor(n_neighbors = 3)
        if classifier == "KNN5" and len(f_train) >= 5:
            clf = KNeighborsRegressor(n_neighbors = 5)
        if classifier == "KNN5" and len(f_train) < 5:
            continue
        if classifier == "KNN7" and len(f_train) >= 7:
            clf = KNeighborsRegressor(n_neighbors = 7)
        if classifier == "KNN7" and len(f_train) < 7:
            continue
        if classifier == "KNN9" and len(f_train) >= 9:
            clf = KNeighborsRegressor(n_neighbors = 9)
        if classifier == "KNN9" and len(f_train) < 9:
            continue
        if classifier == "LR":
            from warnings import filterwarnings
            filterwarnings('ignore')
            clf = LogisticRegression(max_iter = 10000)
        if classifier == "RF":
            clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1, max_features = "sqrt", max_depth = 5)

        clf.fit(np.array(f_train), np.array(train_class))

        if classifier == "LR":
            outcomes2 = pd.concat([outcomes2, pd.DataFrame({"predicted" : clf.predict_proba(f_test)[:,1],
                                                "Class" : test_class,
                                                "chromosome" : chrom,
                                                "nodeID" : f_test.index})])
        else:
            outcomes2 = pd.concat([outcomes2, pd.DataFrame({"predicted" : clf.predict(f_test),
                                                            "Class" : test_class,
                                                            "chromosome" : chrom,
                                                            "nodeID" : f_test.index})])


    # In[16]:


    #outcomes.to_csv("/Users/vlietstraw/git/Post-GWAS/Node2vec/Leave-SNP-Out cross validation " + class_type + " " + str(max_bp_distance) + ".csv", index = False)
    #outcomes2.to_csv("/Users/vlietstraw/git/Post-GWAS/Node2vec/Leave-chromosome-Out cross validation " + class_type + " " + str(max_bp_distance) + ".csv", index = False)


    # ## Evaluate leave-SNP-out

    # In[17]:


    # outcomes = outcomes.sort_values(["SNP ID", "predicted"], ascending = False)
    # outcomes["For-SNP rank"] = outcomes.groupby("SNP ID").cumcount() + 1


    # In[20]:


    #
    #
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes["Class"], -outcomes["For-SNP rank"], pos_label = 1)
    # all_metrics.at[am_index, "ROC-AUC overall (lso)"] = sklearn.metrics.auc(fpr, tpr) * 100
    #
    #
    # # In[21]:
    #
    #
    # # Calculate the ROC-AUC for every SNP and average the result
    # SNPS2 = list(set(outcomes["SNP ID"]))
    # aucs = []
    # for snp in SNPS2:
    #   if len(set(outcomes["Class"][outcomes["SNP ID"] == snp])) == 1:
    #       aucs.append(list(set(outcomes["Class"][outcomes["SNP ID"] == snp]))[0])
    #   else:
    #       fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes["Class"][outcomes["SNP ID"] == snp], -outcomes["For-SNP rank"][outcomes["SNP ID"] == snp], pos_label = 1)
    #       aucs.append(sklearn.metrics.auc(fpr, tpr))
    # all_metrics.at[am_index, "ROC-AUC - mean per snpl (lso)"] = sum(aucs)/len(aucs)
    #
    #
    # # In[22]:
    #
    #
    # # Calculate hits @1
    # all_metrics.at[am_index, "Hits@1(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] == 1)])
    #
    #
    # # In[23]:
    #
    #
    # # Calculate hits @3
    # all_metrics.at[am_index, "Hits@3(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 3)])
    #
    #
    # # In[24]:
    #
    #
    # # Calculate hits @5
    # all_metrics.at[am_index, "Hits@5(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 5)])
    #
    #
    # # In[25]:
    #
    #
    # # Calculate hits @10
    # all_metrics.at[am_index, "Hits@10(lso)"] = sum(outcomes["Class"][(outcomes["Class"] == 1) & (outcomes["For-SNP rank"] <= 10)])
    #
    #
    # # In[26]:
    #
    #
    # all_metrics.at[am_index, "Mean rank (lso)"] = outcomes["For-SNP rank"][(outcomes["Class"] == 1)].mean()
    #
    #
    # # In[27]:
    #
    #
    # all_metrics.at[am_index, "Median rank (lso)"] = outcomes["For-SNP rank"][outcomes["Class"] == 1].quantile(q = [0,0.25,0.5,0.75,1])[0.50]


    # ## Evaluate leave-chromosome-out

    # In[28]:

    if len(outcomes2) > 0:
        outcomes2 = outcomes2.sort_values(["chromosome", "predicted"], ascending = False)
        outcomes2["For-chromosome rank"] = outcomes2.groupby("chromosome").cumcount() + 1


        # In[29]:


        chromosomes = list(set(outcomes2["chromosome"]))
        aucs = []
        for chrom in chromosomes:
          fpr, tpr, thresholds = sklearn.metrics.roc_curve(outcomes2["Class"][outcomes2["chromosome"] == chrom], -outcomes2["For-chromosome rank"][outcomes2["chromosome"] == chrom], pos_label = 1)
          aucs.append(sklearn.metrics.auc(fpr, tpr))

        all_metrics.at[am_index, "ROC-AUC per chromosome"] = sum(aucs)/len(aucs)

        # In[30]:


        ref = ref.merge(outcomes2[["nodeID", "predicted"]], on = "nodeID", how = "left")

        all_metrics.at[am_index, "Recall snps"] = len(set(ref["SNP ID"]))
        all_metrics.at[am_index, "Recall entries"] = sum(ref["Class"])
        all_metrics.at[am_index, "Recall genes"] = len(set(ref["nodeID"][ref["Class"] == 1]))

        ref = ref.sort_values(["SNP ID", "predicted"], ascending = False)

        SNP_temp = 0
        counter = 0
        prediction_temp = 9999
        for indx, row in ref.iterrows():
            if SNP_temp != row["SNP ID"]:
                SNP_temp = row["SNP ID"]
                counter = 1
                prediction_temp = row["predicted"]
            elif SNP_temp == row["SNP ID"] and prediction_temp != row["predicted"]:
                counter += 1
                prediction_temp = row["predicted"]
            ref.at[indx, "For-SNP rank"] = counter
        
        if writeOut:
            ref_out = ref[["SNP ID", "nodeID", "predicted", "Class", "For-SNP rank"]]
            ref_out["For-SNP rank"] = ref_out["For-SNP rank"].astype(int)
            ref_out.to_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/" + am_values["refset"] + " " + am_values["variation"] + " " + am_values["algorithm"] + " " + am_values["bp distance"] + " " + datetime.today().strftime("%d-%m-%Y") + ".csv", sep = ";", index = False)

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


all_metrics.to_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/all_variations_performance_metrics " + datetime.today().strftime("%d-%m-%Y") + ".csv", sep = ";", decimal = ",", index = False)
