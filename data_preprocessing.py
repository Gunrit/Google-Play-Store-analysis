import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# STEP 1: LOAD DATA
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')

print("Files loaded successfully")
print("Apps data shape:", apps_df.shape)
print("Reviews data shape:", reviews_df.shape)


# STEP 2: CLEAN APPS DATA

# Remove rows with missing Rating
apps_df = apps_df.dropna(subset=['Rating'])

# Fill missing values with mode
for column in apps_df.columns:
    apps_df[column] = apps_df[column].fillna(apps_df[column].mode()[0])

# Remove duplicates
apps_df = apps_df.drop_duplicates()

# Keep only valid ratings
apps_df = apps_df[apps_df['Rating'] <= 5]

print("Apps data basic cleaning done")


# STEP 3: CLEAN REVIEWS DATA

# Remove rows where Translated_Review is missing
reviews_df = reviews_df.dropna(subset=['Translated_Review'])

print("Reviews data basic cleaning done")


# STEP 4: DATA TRANSFORMATION

# Clean Installs column
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '', regex=False)
apps_df['Installs'] = apps_df['Installs'].str.replace('+', '', regex=False)
apps_df['Installs'] = apps_df['Installs'].astype(int)

# Clean Price column
apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=False).astype(float)

# Convert Reviews to integer
apps_df['Reviews'] = apps_df['Reviews'].astype(int)

# Convert Last Updated to datetime and extract year
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])
apps_df['Year'] = apps_df['Last Updated'].dt.year

print("Installs, Price, Reviews, and Date columns cleaned")

# STEP 5: SIZE CONVERSION

def convert_size(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024
    return np.nan

apps_df['Size'] = apps_df['Size'].apply(convert_size)

# Fill remaining Size nulls with median
apps_df['Size'] = apps_df['Size'].fillna(apps_df['Size'].median())

print("Size column converted successfully")


# STEP 6: FEATURE ENGINEERING

# Log transformations
apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])

# Revenue feature
apps_df['Revenue'] = apps_df['Installs'] * apps_df['Price']

print("Feature engineering completed")

# STEP 7: SENTIMENT ANALYSIS

sia = SentimentIntensityAnalyzer()

reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)

print("Sentiment analysis completed")

# STEP 8: MERGE DATA

merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')

print("Data merged successfully")
print("Merged data shape:", merged_df.shape)

# STEP 9: SAVE CLEANED DATA

merged_df.to_csv("cleaned_apps_data.csv", index=False)

print("Final cleaned dataset saved successfully")
print(merged_df.head())
print(merged_df.dtypes)