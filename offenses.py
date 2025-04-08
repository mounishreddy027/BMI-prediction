import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'sentencing.csv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter=';')

# Plot the distribution of offenses
plt.figure(figsize=(10, 6))
data['offense'].value_counts().head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 Offenses Committed by Inmates', fontsize=14)
plt.xlabel('Offense', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
