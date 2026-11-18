
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/creditcard.csv")

print("Total Transactions:", len(df))

print("\nTransaction Distribution:")
print(df["Class"].value_counts())

print("\nTransaction Percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

# Visualize fraud and legitimate transactions
sns.countplot(x="Class", data=df)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()
