import numpy as np
from scipy import stats
import pandas as pd


# Get the original dataframe and transform the features
# according to the rules defined
def transform_dataset(house_prices):

	y = np.log(house_prices['SalePrice'])
	x = house_prices.drop(columns=['SalePrice'])

	categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',
							'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
							'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
							'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure',
							'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
							'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive',
							'SaleType', 'SaleCondition', 'Utilities']

	x[categorical_columns] = x[categorical_columns].astype(str)

	continuous_columns = x.drop(columns=categorical_columns).columns

	for column in continuous_columns:
		k, p = stats.normaltest(x[column])
		if p < 0.05:
			x[column] = np.log(x[column] + 1)
		x[column + 'Square'] = x[column]**2
		x[column + 'Cubic'] = x[column]**3

	x = pd.get_dummies(x, drop_first=True)

	x['LotGrBsmtArea'] = x['GrLivArea'] * x['LotArea'] * x['TotalBsmtSF']
	x['Q406'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2006))
	x['Q407'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2007))
	x['Q408'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2008))
	x['Q409'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2009))
	x['Q410'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2010))

	return x, y


# Get the original dataframe and transform the features
# according to the rules defined
def transform_dataset_test(house_prices):

	x = house_prices

	categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',
							'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
							'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
							'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure',
							'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
							'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive',
							'SaleType', 'SaleCondition', 'Utilities']

	x[categorical_columns] = x[categorical_columns].astype(str)

	continuous_columns = x.drop(columns=categorical_columns).columns

	for column in continuous_columns:
		k, p = stats.normaltest(x[column])
		if p < 0.05:
			x[column] = np.log(x[column] + 1)
		x[column + 'Square'] = x[column]**2
		x[column + 'Cubic'] = x[column]**3

	x['LotGrBsmtArea'] = x['GrLivArea'] * x['LotArea'] * x['TotalBsmtSF']
	x['Q406'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2006))
	x['Q407'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2007))
	x['Q408'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2008))
	x['Q409'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2009))
	x['Q410'] = ((x['MoSold'] == 10) | (x['MoSold'] == 11) | (x['MoSold'] == 12) & (x['YrSold'] == 2010))

	x = pd.get_dummies(x, drop_first=True)

	return x
