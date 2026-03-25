import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    "Area": [1000, 1500, 2000, 1200, 1800],
    "Bedrooms": [2, 3, 4, 2, 3],
    "Bathrooms": [1, 2, 3, 2, 2],
    "Price": [200000, 300000, 400000, 250000, 350000]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Define input and output
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict house price
area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))

prediction = model.predict([[area, bedrooms, bathrooms]])

print("Predicted House Price:", prediction[0])