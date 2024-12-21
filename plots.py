import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target


df.drop(columns='target').hist(bins=20, figsize=(10, 8))
plt.suptitle("Distribution of Features", fontsize=16)
plt.savefig('results/hits.png')
plt.show()


sns.pairplot(df, hue='target', palette='Set2', markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Dataset", fontsize=16)
plt.savefig('results/pairplot.png')
plt.show()



corr = df.drop(columns='target').corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Features", fontsize=16)
plt.savefig('results/cor_m.png')
plt.show()

