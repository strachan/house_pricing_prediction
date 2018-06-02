import numpy as np
from scipy import stats


# Get the original dataframe and transform the features
# according to the rules defined
def transform_dataset(house_prices):

	y = np.log(house_prices['SalePrice'])
	x = house_prices.drop(columns=['SalePrice'])

	# Basement variables
	x['BsmtQualBinary'] = (x['BsmtQual'] == 'Ex') | (x['BsmtQual'] == 'Gd')
	x['BsmtExposureBinary'] = (x['BsmtExposure'] == 'Gd') | (x['BsmtExposure'] == 'Av') | (x['BsmtExposure'] == 'Mn')
	x.drop(columns=['BsmtQual', 'BsmtExposure'], inplace=True)

	# Fireplace variables
	x['FireplaceQualBinary'] = (x['FireplaceQu'] == 'Ex') | (x['FireplaceQu'] == 'Gd') | (x['FireplaceQu'] == 'TA')
	x.drop(columns=['FireplaceQu'], inplace=True)

	# Garage variables
	x['GarageTypeBinary'] = (x['GarageType'] == 'Attchd') | (x['GarageType'] == 'BuiltIn')
	x['GarageFinishBinary'] = (x['GarageFinish'] == 'Fin') | (x['GarageFinish'] == 'RFn')
	x.drop(columns=['GarageType', 'GarageFinish'], inplace=True)

	# Electrical variables
	x['ElectricalSBrkr'] = x['Electrical'] == 'SBrkr'
	x.drop(columns=['Electrical'], inplace=True)

	# Masonry veneer variables
	x['MasVnrTypeBinary'] = (x['MasVnrType'] == 'BrkFace') | (x['MasVnrType'] == 'Stone')
	x.drop(columns=['MasVnrType'], inplace=True)

	# Paved Drive variable
	x['PavedDriveBinary'] = x['PavedDrive'] == 'Y'
	x.drop(columns=['PavedDrive'], inplace=True)

	# Sale condition variables
	x['SaleConditionBinary1'] = x['SaleCondition'] == 'Partial'
	x['SaleConditionBinary2'] = (x['SaleCondition'] == 'Abnorml') | (x['SaleCondition'] == 'AdjLand')
	x.drop(columns=['SaleCondition'], inplace=True)

	# Sale type variables
	x['SaleTypeBinary1'] = x['SaleType'] == 'New'
	x['SaleTypeBinary2'] = (x['SaleType'] == 'Oth') | (x['SaleType'] == 'ConLD') | (x['SaleType'] == 'COD')
	x.drop(columns=['SaleType'], inplace=True)

	# Kitchen quality variable
	x['KitchenQualBinary1'] = (x['KitchenQual'] == 'Ex') | (x['KitchenQual'] == 'Gd')
	x.drop(columns=['KitchenQual'], inplace=True)

	# Foundation variable
	x['FoundationBinary'] = x['Foundation'] == 'PConc'
	x.drop(columns=['Foundation'], inplace=True)

	# Exterior condition variable
	x['ExterCondBinary'] = (x['ExterCond'] == 'TA') | (x['ExterCond'] == 'Gd') | (x['ExterCond'] == 'Ex')
	x.drop(columns=['ExterCond'], inplace=True)

	# Exterior quality variable
	x['ExterQualBinary'] = (x['ExterQual'] == 'Ex') | (x['ExterQual'] == 'Gd')
	x.drop(columns=['ExterQual'], inplace=True)

	# General zoning classification of sale variables
	x['MSZoningBinary1'] = x['MSZoning'] == 'FV'
	x['MSZoningBinary2'] = (x['MSZoning'] == 'C (all)') | (x['MSZoning'] == 'RH') | (x['MSZoning'] == 'RM')
	x.drop(columns=['MSZoning'], inplace=True)

	# Roof style variable
	x['RoofStyleBinary'] = x['RoofStyle'] == 'Hip'
	x.drop(columns=['RoofStyle'], inplace=True)

	# Overall quality variable
	x['OverallQualBinary'] = x['OverallQual'].apply(lambda x: int(x) > 5)
	x.drop(columns=['OverallQual'], inplace=True)

	# Neighborhood variables
	x['NeighborhoodPoor'] = ((x['Neighborhood'] == 'BrDale') | (x['Neighborhood'] == 'BrkSide') | (x['Neighborhood'] == 'Edwards') |
							(x['Neighborhood'] == 'IDOTRR') | (x['Neighborhood'] == 'MeadowV') | (x['Neighborhood'] == 'Names') |
							(x['Neighborhood'] == 'OldTown'))
	x['NeighborhoodRich'] = ((x['Neighborhood'] == 'Blmngtn') | (x['Neighborhood'] == 'CollgCr') | (x['Neighborhood'] == 'Gilbirt') |
							(x['Neighborhood'] == 'NoRidge') | (x['Neighborhood'] == 'NWAmes') | (x['Neighborhood'] == 'NridgHt') |
							(x['Neighborhood'] == 'Somerst') | (x['Neighborhood'] == 'StoneBr') | (x['Neighborhood'] == 'Timber') |
							(x['Neighborhood'] == 'Veenker'))
	x.drop(columns=['Neighborhood'], inplace=True)

	# Type of dwelling involved in the sale variables
	x['MSSubClassBinary1'] = ((x['MSSubClass'] == 30) | (x['MSSubClass'] == 45) | (x['MSSubClass'] == 50) |
								(x['MSSubClass'] == 90) | (x['MSSubClass'] == 180) | (x['MSSubClass'] == 190))
	x['MSSubClassBinary2'] = (x['MSSubClass'] == 60) | (x['MSSubClass'] == 120)
	x.drop(columns=['MSSubClass'], inplace=True)

	# Overall condition variable
	x['OverallCondBinary'] = x['OverallCond'] > 4
	x.drop(columns=['OverallCond'], inplace=True)

	# House style variables
	x['HouseStyleBinary1'] = (x['HouseStyle'] == '1.5Fin') | (x['HouseStyle'] == '1.5Unf')
	x['HouseStyleBinary2'] = (x['HouseStyle'] == '2Story')
	x.drop(columns=['HouseStyle'], inplace=True)

	# Exterior covering on house variable
	x['Exterior1stBinary1'] = ((x['Exterior1st'] == 'AsbShng') | (x['Exterior1st'] == 'AsphShn') | (x['Exterior1st'] == 'BrkComm') |
								(x['Exterior1st'] == 'CBlock') | (x['Exterior1st'] == 'MetalSd') | (x['Exterior1st'] == 'Wd Sdng') |
								(x['Exterior1st'] == 'WdShing'))
	x['Exterior1stBinary2'] = (x['Exterior1st'] == 'ImStucc') | (x['Exterior1st'] == 'Stone') | (x['Exterior1st'] == 'VinylSd')
	x.drop(columns=['Exterior1st'], inplace=True)

	# Lot variables
	x['LotShapeBinary'] = x['LotShape'] == 'Reg'
	x['LotConfigBinary'] = x['LotConfig'] == 'CulDSac'
	x.drop(columns=['LotShape', 'LotConfig'], inplace=True)

	# Basement variable
	x['BsmtFinType1Binary'] = x['BsmtFinType1'] == 'GLQ'
	x.drop(columns=['BsmtFinType1'], inplace=True)

	# Heating/Central Air variables
	x['HeatingQCBinary'] = x['HeatingQC'] == 'Ex'
	x['CentralAirBinary'] = x['CentralAir'] == 'N'
	x.drop(columns=['HeatingQC', 'CentralAir'], inplace=True)
	x.drop(columns=['BsmtFullBath', 'BsmtHalfBath'], inplace=True)

	# Drop some variables
	x.drop(columns=['Street', 'Condition2', 'Condition1', 'RoofMatl', 'Heating', 'LandSlope', 'LandContour', 'BldgType',
					'Functional', 'Exterior2nd', 'MiscVal', 'Utilities'], inplace=True)

	# Lot size variable
	x['LotArea'] = np.log(x['LotArea'])

	# Ground living area variable
	x['Has2ndFlr'] = x['2ndFlrSF'] > 0
	x.drop(columns=['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], inplace=True)

	# Basement variables
	x.drop(columns=['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], inplace=True)

	# Lot frontage variable
	trans, lam = stats.boxcox(x['LotFrontage'])
	x['LotFrontage'] = trans

	# Garage area variable
	trans, lam = stats.boxcox(x['GarageArea'] + 1)
	x['GarageArea'] = trans

	# Drop some continuous variables
	x.drop(columns=['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'], inplace=True)

	# Continuous countable variables
	x['GarageCarsBinary'] = x['GarageCars'].apply(lambda x: int(x) > 1)
	x['FireplacesBinary'] = x['Fireplaces'].apply(lambda x: int(x) > 0)
	x['TotRmsAbvGrdBinary1'] = x['TotRmsAbvGrd'].apply(lambda x: int(x) > 5 and int(x) <= 7)
	x['TotRmsAbvGrdBinary2'] = x['TotRmsAbvGrd'].apply(lambda x: int(x) > 7)
	x['KitchenAbvGrBinary'] = x['KitchenAbvGr'].apply(lambda x: int(x) == 1)
	# x['FullBathBinary'] = x['FullBath'].apply(lambda x: int(x) > 1)
	# x['HalfBathBinary'] = x['HalfBath'].apply(lambda x: int(x) == 1)
	x['BedroomAbvGrBinary'] = x['BedroomAbvGr'].apply(lambda x: int(x) == 4)
	x.drop(columns=['GarageCars', 'Fireplaces', 'TotRmsAbvGrd', 'KitchenAbvGr',
					'FullBath', 'HalfBath', 'BedroomAbvGr'], inplace=True)

	# Continuous year / age variables
	x['NewHouse'] = x['YearBuilt'] > 1990
	x['OldHouse'] = x['YearBuilt'] < 1980

	x.drop(columns=['YearRemodAdd', 'MoSold', 'YrSold'], inplace=True)

	return x, y




