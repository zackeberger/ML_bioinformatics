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
B = 10000

# Parse out first SNP from genotype array
snp1 = genotypes[:,0]
NUM_SAMPLES = len(snp1)

# Calculate statistic (T1 = NUM_SAMPLES*(PCC)^2)
snp1_corr, _ = pearsonr(phenotypes, snp1)
snp1_stat = NUM_SAMPLES * (snp1_corr**2)

permutation_stat = np.empty(B)

# Track how many permutations have more extreme statistics than the observations
more_extreme_permutations = 0

# Run permutation test
for i in range(B):
    phen_permutation = np.random.permutation(phenotypes)
    corr, _ = pearsonr(phen_permutation, snp1)            # Calculate Pearson Correlation Coefficient
    stat = NUM_SAMPLES * (corr**2)
    permutation_stat[i] = stat                                  # Add statistic to list of observations

    if stat > snp1_stat:
        more_extreme_permutations += 1


# Plot Histogram of test statistic from B permutations
n, bins, patches = plt.hist(x=permutation_stat, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
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
# TODO: THIS IS WRONG!!!???????/
t1_p_value_permutations = more_extreme_permutations / B
print("T1 p-value w/ permutations: " + str(t1_p_value_permutations))


########################
## Chi2 Approximation ##
########################

# Plot Chi Squared (1 degree of freedom)
x = np.arange(0, .30, .3 / B)
plt.plot(x, chi2.pdf(x, df=1), color='r', lw=2)
# TODO: Appropriate? P  value correct?
#df = 1
#x = np.linspace(0, .3, B)
#plt.plot(x, chi2.pdf(x, df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.axvline(x=snp1_stat, color='magenta', label='Observed T1')
#plt.title('Chi Squared PDF with 1 Degree of Freedom')
plt.show()

# TODO: separate plot
# p-value of T1 based on chi square approximation
t1_p_value_chi2 = 1 - chi2.cdf(snp1_stat, 1)
print("T1 p-value w/ chi squared approximation: " + str(t1_p_value_chi2))
#TODO: NO -2log, because T is chi squared!!!

######################
## Controlling FWER ##
######################

# Use Bonferroni Procedure -- level/NUM_HYPOTHESES
level = 0.05
t = 0.05 / 10
print("Threshold t using Bonferroni Procedure: " + str(t))

# Decide which SNPs to reject with Bonferroni
print("Bonferroni")
for i in range(10):
    snp_corr, _ = pearsonr(phenotypes, genotypes[:,i])
    snp_stat = NUM_SAMPLES*(snp_corr**2)
    p = 1 - chi2.cdf(snp_stat, 1)

    if p <= t:
        print("Reject SNP " + str(i))

# Do simulation for 10,000 times
# Get minimum p value from each permutation
# Sort minimum p values
# Find p value at 95th percent (or 5th depending on order)

# TODO: THROUGHOUT EVERYTHING, SHUFFLE PHENOTYPES NOT GENOTYPES!!!!!
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

        min_p_values[j] = np.min(current_p_values)

    # Sort the minimum p values
    sorted_min_p = np.sort(min_p_values)
    
    # Return the p value such that 95 percent of p values are larger
    # This value is the new threshold
    return sorted_min_p[int(B * level)]


# Use Permutation Testing
B = 10000
level = 0.05

t = FWER_Control(B, level)
print("Threshold t using Permutation Testing: " + str(t))

# Decide which SNPs to reject
print("Permutation")
for i in range(10):
    snp_corr, _ = pearsonr(phenotypes, genotypes[:,i])
    snp_stat = NUM_SAMPLES*(snp_corr**2)
    p = 1 - chi2.cdf(snp_stat, 1)

    if p <= t:
        print("Reject SNP " + str(i))

# TODO: CHI^2 MUST BE NATURAL LOG!!!!!!!
