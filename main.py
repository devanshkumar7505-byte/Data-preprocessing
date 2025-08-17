import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv('rain-agriculture.csv')
print('original_data')
print(df.head())

#step_1: handle missing values
df_cleaned=df.dropna()

#step_2: Fill missing values with column mean
df_filled=df.fillna(df.mean(numeric_only=True))

#step_3:normalize numeric dat with min _max scaling
scaler=MinMaxScaler()

# select only numeric column for scaling
numeric_cols=df_filled.select_dtypes(include=['float64','int64']).columns
df_filled[numeric_cols]=scaler.fit_transform(df_filled[numeric_cols])

#step_4: save the cleaned dataset to a new file
df_filled.to_csv('cleaned_data.csv',index=False)
print("\nCleaned and Normalized Data.")
print(df_filled.head())
print("\nCleaned data saved to 'cleaned_data.csv'")
