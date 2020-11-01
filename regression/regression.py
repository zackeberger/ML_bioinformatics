import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from math import log

# Import data
phenotypes = np.loadtxt("./data/logistic_regression_data/logistic_regression.pheno", dtype='int')
genotype_strings = np.loadtxt("./data/logistic_regression_data/logistic_regression.geno", dtype='str')

# Constants related to data
NUM_SNPS = len(genotype_strings)
NUM_WEIGHTS = NUM_SNPS + 1          # 1 for each SNP plus a bias
NUM_DATA_POINTS = len(phenotypes)
ITERATIONS_TO_TEST = 50
step_sizes = np.zeros(5)            # Step sizes to test
for i in range(5):
    step_sizes[i] = 10**(-(i+1))

# Massage the genotypes into a matrix
genotypes = np.zeros((NUM_WEIGHTS, NUM_DATA_POINTS), dtype='int')
genotypes[0,:] = np.ones(NUM_DATA_POINTS, dtype='int')
for i in range(NUM_SNPS):
    genotypes[i+1,:] = np.array([int(s) for s in list(genotype_strings[i])], dtype='int')


# Gradient Descent on negative log likelihood for logistic regression
# Return vector of NLLs where the i-th entry is the NLL at beta after the (i+1)-th iteration
def gradient_descent(step, iterations):
    # Initialize vector of NLLs
    NLLS = [0]*iterations
    
    # Initialize weights = 0
    beta = np.zeros(NUM_WEIGHTS, dtype='int')
    
    # Calculate gradient at beta
    for j in range(iterations):
        grad_at_beta = 0
        for i in range(NUM_DATA_POINTS):
            x_i = np.array(genotypes[:,i])
            y_i = phenotypes[i]
            sigmoid_at_beta = expit(np.dot(beta, x_i))
            grad_at_beta += (y_i - sigmoid_at_beta)*x_i
       
        # Perform t-th update
        beta = beta + (step * grad_at_beta)
        
        # Calculate NLL at beta
        NLL_j = 1
        for i in range(NUM_DATA_POINTS):
            x_i = np.array(genotypes[:,i])
            y_i = phenotypes[i]
            NLL_j *= expit(np.dot(beta, x_i))**y_i * (1-expit(np.dot(beta, x_i)))**(1-y_i)
        
        if NLL_j == 0:
            NLLS[j] = 0
        else:
            NLLS[j] = -1*log(NLL_j)

    # Return the vector of NLLs
    return NLLS


# Test driver for gradient descent
iterations = list(range(1, 51))
for step_size in step_sizes:
    NLLS = gradient_descent(step_size, ITERATIONS_TO_TEST)

    plt.plot(iterations, NLLS)
    plt.title("NLL vs. Iteration: Step size = " + str(step_size))
    plt.xlabel("Iteration of Gradient Descent")
    plt.ylabel("Negative Log Likelihood for Logistic Regression")
    plt.show()
