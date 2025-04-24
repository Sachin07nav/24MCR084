from pandas import read_csv
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load dataset
filename = "Iris.csv"
data = read_csv(filename)

# Step 2: Display data shape and preview
print("Shape of the dataset:", data.shape)
print("First 20 rows:\n", data.head(20))

# Step 3: Plot and save histograms silently
data.hist()
pyplot.savefig("histograms.png")
pyplot.close()

# Step 4: Plot and save density plots silently
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.savefig("density_plots.png")
pyplot.close()

# Step 5: Convert to NumPy array and extract features/labels
# 🛠 FIX: Ensure you skip the ID column (usually the first one), and handle headers correctly
# If 'Iris.csv' has a 'Species' column, use it directly
X = data.iloc[:, 1:5].values   # Skip ID, select features
Y = data.iloc[:, 5].values     #
