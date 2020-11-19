import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('../data/pca_data/pca.data', sep='\t')

# Look at the data frame
# pd.set_option("display.max_rows", None)
# print(pca_df)

# snps is 1092x13237, each row an individual's genotype
# populations is 1x1092, true populations of each individual
snps = df.filter(regex="rs").to_numpy() 
populations = df['population'].to_numpy()

# Set random seed so that random values are deterministic
np.random.seed(0)

# TODO
