from main import credit_card_data

# first 5 rows of the dataset
credit_card_data.head()

# last 5 rows of the dataset
credit_card_data.tail()

# information of the dataset
credit_card_data.info()

# checks the number of missing values in each column displayed in info
credit_card_data.isnull().sum()

# distribution of legit (0) and fraudulent (1) transactions
credit_card_data['Class'].value_counts()

# statistical measures of the data
credit_card_data[credit_card_data.Class == 0].Amount.describe()

# compare values for both transactions
credit_card_data.groupby('Class').mean()