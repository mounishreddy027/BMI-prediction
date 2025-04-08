import pandas as pd

# Load the CSV file
df = pd.read_csv('person.csv', delimiter=';')

# Convert weight (pounds to kilograms) and height (inches to meters)
df['Weight_kg'] = df['weight'] * 0.453592  # 1 pound = 0.453592 kg
df['Height_m'] = df['height'] * 0.0254    # 1 inch = 0.0254 meters

# Calculate BMI
df['BMI'] = df['Weight_kg'] / (df['Height_m'] ** 2)

# Categorize BMI
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    else:
        return 'Overweight'
    

df['Category'] = df['BMI'].apply(bmi_category)

# Create a new dataset with only the ID and Category columns
bmi_df = df[['id', 'Category']]

# Use only the first 10,000 rows
bmi_df = bmi_df.head(10000)

# Save the new dataset
bmi_df.to_csv('bmi.csv', index=False)

print("BMI categories dataset (first 10,000 rows) created successfully!")
