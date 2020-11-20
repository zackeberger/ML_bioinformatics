import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import data
df = pd.read_csv('../data/pca_data/pca.data', sep='\t')

# snps (1092x13237)    --  each row an individual's genotype
# populations (1x1092) -- true populations of each individual
snps = df.filter(regex="rs").to_numpy() 
population_assignments = df['population'].to_numpy()

# Set random seed so that random values are deterministic
np.random.seed(0)


# We will aim to generate clusters of genomic data, where each cluster represents 
# an ancestral population using two different approaches
# 1 -- assign clusters based on true population label
# 2 -- assign clusters with k-means clustering


# Run KMeans  to obtain clusters
kmeans = KMeans(n_clusters=4, n_init=5)
kmeans.fit(snps)
cluster_assignments = kmeans.labels_


# Run PCA two obtain first 2 principle components of snp data
pca = PCA(n_components=2)
pca.fit(snps)
reduced_snps = pca.transform(snps)

# African (AFR) -- Green    East Asian (ASN) -- Yellow
# American (AMR) -- Red     European (EUR) -- Blue
pop_colors = {'AFR': 'g', 'ASN': 'y', 'AMR': 'r', 'EUR': 'b'}   # Map true population label to color
cluster_colors = {0: 'b', 1: 'y', 2: 'g', 3: 'r'}               # Map cluster assignment to color

# Generate plot of data
def make_graph(assignments, colors, reduced_snps):

    for i in range(len(assignments)):
        a = assignments[i]
        col = colors[a]
        pc = reduced_snps[i]
        plt.scatter(pc[0], pc[1], c=col)

    plt.title("PC1 vs. PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# Plot PC1 vs. PC2 for each datapoint, using pop_colors
#make_graph(population_assignments, pop_colors, reduced_snps)

# Plot PC1 vs. PC2 for each datapoint, using cluster_colors
#make_graph(cluster_assignments, cluster_colors, reduced_snps)


# Calculate fraction of cluster assignments that agree with the true population labels
total = len(population_assignments)
agree = 0
for i in range(total):
    if pop_colors[population_assignments[i]] == cluster_colors[cluster_assignments[i]]:
        agree += 1

print("Fraction of cluster assignments that agree with true label: " + str(agree) + "/" + str(total)  + " = " + str(agree / total))
