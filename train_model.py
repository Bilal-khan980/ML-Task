import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Generate random data for plot size (in square feet) and price (in thousands of dollars)
np.random.seed(42)
plot_sizes = np.random.randint(500, 5000, 100)  # Random sizes between 500 and 5000 sqft
prices = plot_sizes * 0.3 + np.random.normal(0, 30, 100)  # Price based on size with some noise

# Create a DataFrame
df = pd.DataFrame({'plot_size': plot_sizes, 'price': prices})

# Split into training and test sets
X = df[['plot_size']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
