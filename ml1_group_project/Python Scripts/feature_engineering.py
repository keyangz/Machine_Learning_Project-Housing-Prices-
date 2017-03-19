import numpy as np
import pandas as pd
from scipy.stats import skew

def transform_features(df):

    ##### Neighborhood #####
    # I mainly created this to for interaction terms (see the function below).
    # I still haven't figured out a good grouping for the more expensive neighborhoods.
    # The idea is that a "really nice" home synergizes well with also being in a good neighborhood.
    good_neigh = set(['Noridge', 'NridgHt', 'NoRidge', 'StoneBr'])
    df['Neighborhood_Good'] = [1 if val in good_neigh else 0 for val in df['Neighborhood']]

    ##### MSSubClass #####
    # Convert to string, since it's a code
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    regroup = []
    for val in df['MSSubClass']:
        if val == '160':
            regroup.append(val)
        else:
            regroup.append('Other')
    df['MSSubClass_Regrouped'] = pd.Series(regroup)
    df = df.drop(['MSSubClass'], 1)

    ##### MSZoning #####
    # Drop this categorical variable - it has small number of observations and is adding noise to LASSO.
    df.loc[df['MSZoning'] == 'C (all)', 'MSZoning'] = np.NaN

    ##### SaleCondition #####
    df.loc[df['SaleCondition'] == 'Family', 'SaleCondition'] = np.NaN

    ##### GarageType #####
    df.loc[df['GarageType'] == '2Types', 'GarageType'] = np.NaN

    ##### Alley #####
    # Was adding noise to the data since most houses don't have an alley
    df = df.drop(['Alley'], 1)

    ##### Functional #####
    # Drop Maj2,Sev since it has only 5,1 observations. This is done in main.py. Also drop
    #df.loc[df['Functional'] == 'Maj2', 'Function'] = np.NaN

    ##### OverallCond #####
    # Convert to string, since it doesn't look linearly related to price
    # For some reason, this was adding noise when we treated it as a categorical.
    # Leaving it as a numeric seems to lead to better MSE.
    # We could explore a non-linear spacing of the values (e.g., quadratic).
    # df['OverallCond'] = df['OverallCond'].astype(str)

    ##### LotFrontage #####
    # Impute missing value with mean
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
    # Add squared term to account for non-linearity
    df['sqLotFrontage'] = df['LotFrontage'] ** 2

    ##### LotArea #####
    # Add square-root term based on shape of date wrt response
    df['sqrtLotArea'] = np.sqrt(df['LotArea'])

    ##### YearBuilt #####
    # Add square term
    df['sqYearBuilt'] = df['YearBuilt'] ** 2

    ##### MasVnrArea #####
    # Impute missing values for MasVnrArea
    df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

    ##### BsmtFinSF1 and BsmtFinSF2 #####
    # Add squared term to account for no basement case non-linearity
    df['sqBsmtFinSF1'] = df['BsmtFinSF1'] ** 2
    df['sqBsmtFinSF2'] = df['BsmtFinSF2'] ** 2

    ##### BsmtUnfSF #####
    # Add squared term to account for no basement case non-linearity
    df['sqBsmtUnfSF'] = df['BsmtUnfSF'] ** 2

    ##### GrLivArea #####
    # Add squared term for non-linearity.
    df['GrLivArea_sq'] = df['GrLivArea'] ** 2

    ##### FullBath #####
    # Impute 0 values with mean
    df['FullBath'] = np.where((df['FullBath'] == 0), df['FullBath'].mean(), df['FullBath'])

    ##### BedroomAbvGr #####
    # Impute 0 values with mean
    df['BedroomAbvGr'] = np.where((df['BedroomAbvGr']==0), df['BedroomAbvGr'].mean(), df['BedroomAbvGr'])

    ##### add square term for garage year built
    df['sqGarageYrBlt'] = df['GarageYrBlt'] ** 2

    ##### add square term of Garage Area #####
    df['sqGarageArea'] = df['GarageArea'] ** 2

    ##### add square term for WoodDeckSF #####
    df['sqWoodDeckSF'] = df['WoodDeckSF'] ** 2

    ##### add square term for OpenPorchSF #####
    df['sqOpenPorchSF'] = df['OpenPorchSF'] ** 2

    ##### add square term for EnclosedPorch #####
    df['sqEnclosedPorch'] = df['EnclosedPorch'] ** 2

    ##### add square term for ScreenPorch #####
    df['sqScreenPorch'] = df['ScreenPorch'] ** 2

    ##### add square term for 3SsnPorch #####
    df['sq3SsnPorch'] = df['3SsnPorch'] ** 2

    ##### add square term for PoolArea #####
    df = df.drop(['PoolArea'], 1)

    ##### PoolQC #####
    df['HasPool'] = pd.notnull(df['PoolQC']).astype('int')
    # Drop this column, there's not enough data in any category
    df = df.drop(['PoolQC'], 1)

    ##### Id #####
    # Drop this column, it's just an identifier
    df = df.drop(['Id'], 1)

    ##### Heating ######
    # Slightly drops CV MSE when regrouping to get rid of noisy categories
    # Heating_Grav is noisy and gets a large (absolute) coefficient
    regroup = []
    for val in df['Heating']:
        if val == 'GasA' or val == 'GasW':
            regroup.append('Gas')
        else:
            regroup.append('Other')
    df['Heating_Regrouped'] = pd.Series(regroup)
    df = df.drop(['Heating'], 1)

    ##### Street #####
    # This categorical variable was adding noise to the results.
    # There are only two groupings and most observations (>90%) were in one group.
    df = df.drop(['Street'], 1)

    ##### Condition2 #####
    # All the observations fell into one category.
    # Drop to avoid getting coefficients for the noisy categories.
    df = df.drop(['Condition2'], 1)

    ##### KitchenAbvGr #####
    df.loc[df['KitchenAbvGr'] == 0, 'KitchenAbvGr'] = 1
    df.loc[df['KitchenAbvGr'] == 3, 'KitchenAbvGr'] = 2

    ##### GarageType #####
    # Was exploring regrouping some of these categories
    # df.loc[df['GarageType'] == 'CarPort', 'GarageType'] = 'Other'
    # df.loc[df['GarageType'].isnull(), 'GarageType'] = 'Other'

    ##### GarageQual #####
    # Was exploring regrouping some of these categories
    # df['GarageQual'] = pd.Series(['Good' if val in ['Gd', 'TA', 'Ex'] else 'Bad' for val in df['GarageQual']])

    ##### PavedDrive #####
    # Regrouping this categorical variable actually led to better performance.
    df['PavedDrive'] = pd.Series(['N' if val == 'N' else 'Y' for val in df['PavedDrive']])

    ##### LandSlope #####
    # Try dropping this, it's conflated with neighborhood.
    df = df.drop(['LandSlope'], 1)

    ##### Interactions #####
    df = create_interaction_terms(df)

    # For categorical features, remove any categories with less than 3 values
    categorical_cols = list(df.select_dtypes(include=['object']).columns.values)
    for col in categorical_cols:
        category_to_remove = list(df[col].value_counts()[df[col].value_counts() <= 3].index)
        if category_to_remove:
            df[col] = df[col].replace(category_to_remove, np.nan)

    # For numeric feature, log transform skewed distributions
    numeric_cols = list(df.select_dtypes(exclude=['object']).columns.values)
    for col in numeric_cols:
        if skew(df[col]) > 0.75 or skew(df[col]) < -0.75:
            df[col] = np.log1p(df[col])

    return df



def transform_target(df):
    ##### Response = SalePrice #####
    df['logSalePrice'] = np.log(df['SalePrice'])
    df = df.drop('SalePrice', 1)
    return df


def create_interaction_terms(df):

    # Re-write code later: there is probably a better way to create interaction terms.

    # Good neighborhoods and TotRmsAbvGrd
    ngh_good_interact_rooms = df.loc[:, ['Neighborhood_Good', 'TotRmsAbvGrd']].prod(1)
    ngh_good_interact_rooms.name = 'Neighborhood_Good_TotRmsAbvGrd'
    df = pd.concat((df, ngh_good_interact_rooms), axis=1)

    # GarageQual and GarageArea
    # This may not be a useful interaction term, was exploring to see how useful it is.
    for val in ['Fa', 'Gd', 'TA']:
        col = 'GarageQual_%s' % val
        garage_qual_area = df.loc[:, [col, 'GarageArea']].prod(1)
        garage_qual_area.name = 'GarageQual_%s_Area' % val
        df = pd.concat((df, garage_qual_area), axis=1)

    return df
