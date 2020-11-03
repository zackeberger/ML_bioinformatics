import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import data
phenotypes = np.loadtxt("../data/linear_regression_data/linear_regression.pheno", dtype='float')
genotype_strings = np.loadtxt("../data/linear_regression_data/linear_regression.geno", dtype='str')

# Constants related to data
NUM_PHENOTYPES = len(phenotypes[0])
NUM_SNPS = len(genotype_strings)
NUM_DATA_POINTS = len(phenotypes)
LEVEL = 0.05

# Massage the genotypes into a numpy matrix
# Columns are individuals
genotypes = np.zeros((NUM_SNPS, NUM_DATA_POINTS), dtype='int')
for i in range(NUM_SNPS):
    genotypes[i,:] = np.array([int(s) for s in list(genotype_strings[i])], dtype='int')

# Calculate rejection threshold with Bonferroni Procedure
threshold = LEVEL / NUM_SNPS

# Store p-values from the following regression test
p_values = np.zeros((NUM_PHENOTYPES, NUM_SNPS))

# Note any SNPs associated to phenotypes
associations = {new_list: [] for new_list in range(NUM_PHENOTYPES)} 

# Run the regression test
for i in range(NUM_PHENOTYPES):
    for j in range(NUM_SNPS):
        _, _, _, p_value, _ = stats.linregress(genotypes[j,:], phenotypes[:,i])
        p_values[i,j] = p_value
        if p_value <= threshold:
            print("SNP " + str(j+1) + " is associated to phenotype " + str(i+1))
            associations[i].append(j)

# Plot p-values of SNP tests for each phenotype
# Visually inspect if they are uniformly distributed
for i in range(NUM_PHENOTYPES):
    n, bins, patches = plt.hist(x=p_values[i,:], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.title("p-values for Phenotype " + str(i+1))
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

# Make box plot of associated SNPs
for i in range(NUM_PHENOTYPES):
    for s in associations[i]:
        data = {'Phenotype': phenotypes[:,i], 'Number of Bases': genotypes[s,:]}
        df = pd.DataFrame(data, columns=['Phenotype', 'Number of Bases'])
        boxplot = df.boxplot(by='Number of Bases', column=['Phenotype'])
        plt.title("Phenotype " + str(i+1) + ", SNP " + str(s+1))
        plt.show()
