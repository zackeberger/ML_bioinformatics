import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import chi2

# Import data
genotypes = np.loadtxt(open("./data/ps1.genos", "r"), delimiter=" ")
phenotypes = np.loadtxt(open("./data/ps1.phenos", "r"), delimiter=" ")

######################
## Permutation Test ##
######################

# Hyperparamters
# - B is number of permutations
B = 100000

# Parse out first SNP from genotype array
snp1 = genotypes[:,0]
NUM_SAMPLES = len(snp1)

# Calculate statistic (T1 = NUM_SAMPLES*(PCC)^2)
snp1_corr, _ = pearsonr(phenotypes, snp1)
snp1_stat = NUM_SAMPLES * (snp1_corr**2)

permutation_stats = np.empty(B)

# Track how many permutations have more extreme statistics than the observations
more_extreme_permutations = 0

# Run permutation test
for i in range(B):
    phen_permutation = np.random.permutation(phenotypes)    # Permute the phenotype
    corr, _ = pearsonr(phen_permutation, snp1)              # Calculate Pearson Correlation Coefficient
    stat = NUM_SAMPLES * (corr**2)                          # Calculate test statistic
    permutation_stats[i] = stat                              # Add statistic to list of observations

    # If the permuted stat is greater than the actual T1, note this
    if stat > snp1_stat:
        more_extreme_permutations += 1


# Plot Histogram of test statistic from B permutations
n, bins, patches = plt.hist(x=permutation_stats, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Statistic')
plt.ylabel('Frequency')
plt.title('Statistic Frequency in Permutation Test')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# Plot observed test statistic T1
plt.axvline(x=snp1_stat, color='magenta', label='Observed T1')
plt.show()

# Calculate p-value of T1
t1_p_value_permutations = more_extreme_permutations / B
print("T1 p-value w/ permutations: " + str(t1_p_value_permutations))



########################
## Chi2 Approximation ##
########################

# Plot Chi Squared (1 degree of freedom)
x = np.arange(0, .30, .3 / B)
plt.plot(x, chi2.pdf(x, df=1), color='r', lw=2)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Chi Squared with 1 Degree of Freedom')
plt.show()

# p-value of T1 based on chi square approximation
t1_p_value_chi2 = 1 - chi2.cdf(snp1_stat, 1)
print("T1 p-value w/ chi squared approximation: " + str(t1_p_value_chi2))



######################
## Controlling FWER ##
######################

# Control FWER with Bonferroni Procedure -- level/NUM_HYPOTHESES
level = 0.05
NUM_TESTS = 10
t = level / NUM_TESTS
print("Threshold t using Bonferroni Procedure: " + str(t))

# Decide which SNPs to reject with Bonferroni
# i.e. these are the SNPs associated with the phenotype
print("SNPS to reject with Bonferroni:")
for i in range(10):
    snp_corr, _ = pearsonr(phenotypes, genotypes[:,i])
    snp_stat = NUM_SAMPLES*(snp_corr**2)
    p = 1 - chi2.cdf(snp_stat, 1)

    if p <= t:
        print("Reject SNP " + str(i + 1))

print("------")


# Algorithm to estimate FWER with permutation testing:
# - Take permutations for B times
# - Get minimum p value over each 10 tests from each permutation
# - Sort minimum p values
# - Find p value at (level)th percentile of array 
#
# Then, the chosen p value is such that only (level)th tests
# are rejected, as desired


# B is the number of permutations
def FWER_Control(B, level):

    min_p_values = np.empty(B)
    
    # For each permutation, get minimum p value accross all 10 SNPs
    for j in range(B):
    
        current_p_values = np.empty(10)
        phen_permutation = np.random.permutation(phenotypes)
        for i in range(10):
            snp_corr, _ = pearsonr(phen_permutation, genotypes[:,i])    # Calculate PCC
            snp_stat = NUM_SAMPLES*(snp_corr**2)                        # Calculate statistic
            
            current_p_values[i] = 1 - chi2.cdf(snp_stat, 1)             # Calculate p value

        # Store minimum p value
        min_p_values[j] = np.min(current_p_values)

    # Sort the minimum p values
    sorted_min_p = np.sort(min_p_values)
    
    # Return the p value such that(level) percent of p values are larger
    # This value is the new threshold
    return sorted_min_p[int(B * level)]


# Control FWER with Permutation Testing
level = 0.05
t = FWER_Control(B, level)
print("Threshold t using Permutation Testing: " + str(t))

# Decide which SNPs to reject
print("SNPS to reject with Permutation Test:")
for i in range(10):
    snp_corr, _ = pearsonr(phenotypes, genotypes[:,i])
    snp_stat = NUM_SAMPLES*(snp_corr**2)
    p = 1 - chi2.cdf(snp_stat, 1)

    if p <= t:
        print("Reject SNP " + str(i + 1))
