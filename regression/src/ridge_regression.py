import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import data
phenotypes_train = np.loadtxt("../data/ridge_regression_data/ridge_training.pheno", dtype='float')
genotype_strings_train = np.loadtxt("../data/ridge_regression_data/ridge_training.geno", dtype='str')
phenotypes_test = np.loadtxt("../data/ridge_regression_data/ridge_test.pheno", dtype='float')
genotype_strings_test = np.loadtxt("../data/ridge_regression_data/ridge_test.geno", dtype='str')

# Constants related to data
NUM_PHENOTYPES_TRAIN = len(phenotypes_train)
NUM_DATA_TRAIN = NUM_PHENOTYPES_TRAIN
NUM_SNPS_TRAIN = len(genotype_strings_train)
NUM_PHENOTYPES_TEST = len(phenotypes_test)
NUM_DATA_TEST = NUM_PHENOTYPES_TEST
NUM_SNPS_TEST = len(genotype_strings_test)

print("Ridge Regression on")
print(" - Training Set: " + str(NUM_DATA_TRAIN) + " individuals at " + str(NUM_SNPS_TRAIN) + " SNPS")
print(" - Testing Set: " + str(NUM_DATA_TEST) + " individuals at " + str(NUM_SNPS_TEST) + " SNPS")

# Massage the genotypes into a numpy matrix
# Columns are individuals
genotypes_train = np.zeros((NUM_SNPS_TRAIN, NUM_DATA_TRAIN), dtype='int')
for i in range(NUM_SNPS_TRAIN):
    genotypes_train[i,:] = np.array([int(s) for s in list(genotype_strings_train[i])], dtype='int')

genotypes_test = np.zeros((NUM_SNPS_TEST, NUM_DATA_TEST), dtype='int')
for i in range(NUM_SNPS_TEST):
    genotypes_test[i,:] = np.array([int(s) for s in list(genotype_strings_test[i])], dtype='int')


# Suppose X has feature vectors on columns
def ridge_regression(X, y, lamb):
    lambI = np.zeros((X.shape[0], X.shape[0]))
    np.fill_diagonal(lambI, lamb)
    inv = np.linalg.inv(np.matmul(X, np.transpose(X)) + lambI)

    beta = np.matmul(inv, np.matmul(X, y))
    return beta

# Run ridge regression for lambda in {2, 5, 8}
beta_lamb2 = ridge_regression(genotypes_train, phenotypes_train, 2)
beta_lamb5 = ridge_regression(genotypes_train, phenotypes_train, 5)
beta_lamb8 = ridge_regression(genotypes_train, phenotypes_train, 8)

# Find predictions on test data
phenotypes_predict_lamb2 = np.matmul(np.transpose(genotypes_test), beta_lamb2)
phenotypes_predict_lamb5 = np.matmul(np.transpose(genotypes_test), beta_lamb5)
phenotypes_predict_lamb8 = np.matmul(np.transpose(genotypes_test), beta_lamb8)

# Calculate MSE
mse_lamb2 = (1 / NUM_DATA_TEST) * np.sum( (phenotypes_test - phenotypes_predict_lamb2)**2 ) 
mse_lamb5 = (1 / NUM_DATA_TEST) * np.sum( (phenotypes_test - phenotypes_predict_lamb5)**2 )
mse_lamb8 = (1 / NUM_DATA_TEST) * np.sum( (phenotypes_test - phenotypes_predict_lamb8)**2 )

# Plot MSE with testing data for lambda in {2, 5, 8}
mse = [mse_lamb2, mse_lamb5, mse_lamb8]
param = [2, 5, 8]
plt.scatter(param, mse)
plt.title("MSE vs. Parameter Setting λ")
plt.xlabel("Parameter Setting λ")
plt.ylabel("MSE")
plt.show()
