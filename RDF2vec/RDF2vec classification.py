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
variations = ["normal", "autoencode", "combi"] #"graphlet",
all_bp_distances = [100, 500, 1000, 2000]
refsets = ["Teslovich", "DeRycke", "farashi", "farashi p-value cutoff"]

all_metrics = pd.DataFrame(list(product(refsets, ML_algorithms, variations, all_bp_distances)), columns = ["refset", "algorithm", "variation", "bp distance"])

for am_index, am_values in all_metrics.iterrows():
    print("Predicting row " + str(am_index) + " of " + str(len(all_metrics)))
    # In[2]:

    classifier = am_values["algorithm"]
    class_type = am_values["variation"]


    # Load the reference set
    if am_values["refset"] == "farashi":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Farashi/Farashi full 2000000 bp distance no pvalue filtering.csv")

    if am_values["refset"] == "farashi p-value cutoff":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Farashi/Farashi full 2000000 bp distance no pvalue filtering.csv")
        ref = ref[ref["GWAS/eQTL p-value¥"] <= float("5e-8")]

    if am_values["refset"] == "DeRycke":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/DeRycke/DeRycke reference set.csv")
        ref.columns = ["SNP ID", "chromosome", "location", "gene_ids", "gene name", "gene start", "gene stop", "Class", "bp distance absolute", "bp distance", "Gene rank"]

        if class_type == "combi":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_derycke_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/DeRycke complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)

        # Extend embeddings with graphlets
        if class_type == "graphlet":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/DeRycke complete protein-protein embeddings.csv", index_col = 0)
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)

        # Load the data
        if class_type == "normal":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/DeRycke complete protein-protein embeddings.csv", index_col = 0)

        # Load the autoencoded embeddings
        if class_type == "autoencode":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_derycke_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/DeRycke complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index

    if am_values["refset"] == "Teslovich":
        ref = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/Input sets/Teslovich/Teslovich reference set.csv")
        ref.columns = ["SNP ID", "chromosome", "location", "P", "gene_ids", "gene name", "gene start", "gene stop", "Class", "bp distance absolute", "bp distance", "Gene rank"]

        if class_type == "combi":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_teslovich_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Teslovich complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)

        # Extend embeddings with graphlets
        if class_type == "graphlet":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Teslovich complete protein-protein embeddings.csv", index_col = 0)
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)

        # Load the data
        if class_type == "normal":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Teslovich complete protein-protein embeddings.csv", index_col = 0)

        # Load the autoencoded embeddings
        if class_type == "autoencode":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_teslovich_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Teslovich complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index

    # In[4]:

    if am_values["refset"] == "farashi p-value cutoff" or am_values["refset"] == "farashi":
        if class_type == "combi":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_farashi_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Farashi complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)
            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)


        # In[5]:


        # Extend embeddings with graphlets
        if class_type == "graphlet":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Farashi complete protein-protein embeddings.csv", index_col = 0)
            graphlets = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/EVOKE/unfiltered.txt", header = None, sep = " ", skiprows = 1)

            # Load the mapping file
            with open("/Users/vlietstraw/git/Post-GWAS/ENSEMBL_mappings.json", "r") as fp:
                ensembl_dict = json.load(fp)
            inv_map = {v: k for k, v in ensembl_dict.items()}

            graphlets.drop(columns = [73], inplace = True)
            graphlets = graphlets.apply(lambda x: np.log10(x, where = x > 0))
            graphlets.columns = ["Graphlet " + str(x) for x in range(len(list(graphlets)))]
            graphlets.index = [x + 1 for x in list(graphlets.index)]
            graphlets.index = [inv_map[x] for x in graphlets.index]
            f = f.merge(graphlets, right_index = True, left_index = True)


        # Load the autoencoded embeddings
        if class_type == "autoencode":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/autorcode_farashi_emb.txt", sep = "\t", header = None)
            f.drop(columns = [350], inplace = True)
            f2 = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Farashi complete protein-protein embeddings.csv", index_col = 0)
            f.index = f2.index


        # Load the data
        if class_type == "normal":
            f = pd.read_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Farashi complete protein-protein embeddings.csv", index_col = 0)


    # In[7]:


    # Set bp distance cutoff
    max_bp_distance = am_values["bp distance"]
    max_bp_distance = max_bp_distance * 1000
    ref = ref[ref["bp distance absolute"] <= max_bp_distance]


    # Drop all SNPs which no longer have a positive case
    ref = ref[ref["gene_ids"].isin(f.index)]
    pos_counts = ref.groupby("SNP ID")["Class"].sum()
    ref = ref[~ref["SNP ID"].isin(pos_counts[pos_counts == 0].index)]


    # In[10]:


    # Identify all genes which are at least once positive
    positives = ref.groupby("gene_ids")["Class"].sum()
    positives[positives > 1] = 1

    f = f.merge(positives, left_index = True, right_index = True)


    # ## Leave SNP out classification

    # In[11]:


    outcomes = pd.DataFrame()
    train_auc_score = []
    train_auc_rank = []


    # In[12]:


    # Perform leave-SNP-out cross validation
    SNPs = list(set(ref["SNP ID"]))
    for snp in SNPs:
        print("Predicting candidates for " + snp + ", number " + str(SNPs.index(snp) + 1) + " out of " + str(len(SNPs)))

        f_test = f[f.index.isin(ref[ref["SNP ID"] == snp]["gene_ids"])].copy()
        f_train = f[f.index.isin(ref[ref["SNP ID"] != snp]["gene_ids"])].copy()

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
                                                        "gene_ids" : f_test.index})])

    outcomes = outcomes.merge(ref[["SNP ID", "gene_ids", "Class"]], on = ["SNP ID", "gene_ids"], how = "left")


    # ## Leave chromosome out validation

    # In[13]:


    outcomes2 = pd.DataFrame()
    train_auc_score2 = []
    train_auc_rank2 = []


    # In[14]:


    # Perform leave-SNP-out cross validation
    chromosomes = list(set(ref["chromosome"]))

    for chrom in chromosomes:
        print("Predicting candidates for chromosome " + str(chrom))

        f_test = f[f.index.isin(ref["gene_ids"][ref["chromosome"] == chrom])].copy()
        f_train = f[f.index.isin(ref["gene_ids"][ref["chromosome"] != chrom])].copy()

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
                                                        "gene_ids" : f_test.index})])


    # In[15]:


    #outcomes.to_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Leave-SNP-Out combi graphlets validation " + class_type + " " + str(max_bp_distance) + " " + classifier + ".csv", index = False)
    #outcomes2.to_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/Leave-chromosome-Out graphlets cross validation " + class_type + " " + str(max_bp_distance) + " " + classifier + ".csv" index = False)


    # ## Evaluate leave-SNP-out

    # In[ ]:


    outcomes = outcomes.sort_values(["SNP ID", "predicted"], ascending = False)
    outcomes["For-SNP rank"] = outcomes.groupby("SNP ID").cumcount() + 1


    # In[ ]:

    all_metrics.at[am_index, "Recall snps"] = len(set(outcomes["SNP ID"]))
    all_metrics.at[am_index, "Recall genes"] = sum(outcomes["Class"])


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


    ref = ref.merge(outcomes2[["gene_ids", "predicted"]], on = "gene_ids", how = "left")


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

all_metrics.to_csv("/Users/vlietstraw/git/Post-GWAS/RDF2vec/all_variations_performance_metrics.csv", sep = ";", decimal = ",", index = False)
