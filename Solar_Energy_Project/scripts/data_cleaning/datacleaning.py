# import data cleaning libraries
import numpy as np
import pandas as pd

def clean_data(df):
    
    # read the dataframe
    df = pd.read_csv(df)
    
    # select some features
    
    df = df[['PeriodEnd', 'PeriodStart', 'AirTemp', 'Dhi', 'Dni', 'Ghi',\
       'PrecipitableWater', 'RelativeHumidity', 'SurfacePressure',\
       'WindDirection10m', 'WindSpeed10m']]
    
    # rename columns
    
    old_names = ['PeriodEnd', 'PeriodStart', 'AirTemp', 'Dhi', 'Dni', 'Ghi','PrecipitableWater', 'RelativeHumidity',\
                'SurfacePressure', 'WindDirection10m', 'WindSpeed10m'] 
    new_names = ['PeriodEnd','PeriodStart','Temperature', 'DHI', 'DNI', 'Radiation', 'Precipitation', 'Humidity',\
                'Pressure', 'WindDirection', 'WindSpeed']
    df.rename(columns = dict(zip(old_names, new_names)), inplace = True)

     # interpret columns as appropriate data types to ensure compatibility
    df['Radiation']     = df['Radiation'].astype(float)
    df['Temperature']   = df['Temperature'].astype(float) # or int
    df['Pressure']      = df['Pressure'].astype(float)
    df['Humidity']      = df['Humidity'].astype(int) # or int
    df['WindDirection'] = df['WindDirection'].astype(float)
    df['WindSpeed']     = df['WindSpeed'].astype(float)
    df['PeriodStart'] = pd.to_datetime(df['PeriodStart']).dt.to_period('T').dt.to_timestamp()
    df['PeriodEnd'] = pd.to_datetime(df['PeriodEnd']).dt.to_period('T').dt.to_timestamp()

    # filter day time data while filtering off night time because there is no solar irradiation
    df_new = df.loc[~((df['Radiation'] == 0) & (df['DNI'] == 0) & (df['DHI'] == 0)),:]

    # convert datetime column of endperiod to just date
    df_new['Date'] = pd.to_datetime(df_new['PeriodEnd']).dt.normalize()

    # convert the column (it's a string) to datetime type
    dt_series = pd.to_datetime(df_new['Date'])

    # create datetime index passing the datetime series
    dt_index = pd.DatetimeIndex(dt_series.values)

    # set date as index 
    df_new1 = df_new.set_index(dt_index)

    # sample daily average for the solar and weather features
    df_new1['Daily_radiation'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['Radiation'].mean()
    df_new1['Daily_DNI'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['DNI'].mean()
    df_new1['Daily_DHI'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['DHI'].mean()
    df_new1['Daily_Temp'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['Temperature'].mean()
    df_new1['Daily_Precip'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['Precipitation'].mean()
    df_new1['Daily_Humidity'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['Humidity'].mean()
    df_new1['Daily_Pressure'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['Pressure'].mean()
    df_new1['Daily_WindDir'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['WindDirection'].mean()
    df_new1['Daily_WindSpeed'] = df_new1.reset_index().groupby(pd.Grouper(key='Date', freq='1D'))['WindSpeed'].mean()

    # drop irrelevant features
    new_df = df_new1.drop(['PeriodStart', 'YearPS', 'HourPS', 'HourPE', 'Sunrise', 'Sunset', 'MonthPS', 'YearPE'], axis= 1)

    # dropping ALL duplicate values exceept the last value
    new_df = new_df[~new_df.Date.duplicated(keep = 'last')]

    # select column features
    final_df = new_df[['MonthPE', 'Date','Daily_Temp','Daily_Precip', 'Daily_Humidity', 'Daily_Pressure',\
                  'Daily_WindDir','Daily_WindSpeed','Daily_DNI', 'Daily_DHI','Daily_radiation']]

    # save cleaned dataset
    final_df.to_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/cleaned_solar_irradiation.csv')

    """
    Create training, test, and validation data set
    """

    # Feature Engineering of Time Series Column
    final_df['Date'] = pd.to_datetime(final_df['Date'], format = '%Y-%m-%d')
    final_df['year'] = final_df['Date'].dt.year
    final_df['month'] = final_df['Date'].dt.month
    final_df['day'] = final_df['Date'].dt.day

    # select features
    cleaned_df = final_df[['month', 'day', 'Daily_Temp', 'Daily_Precip', 'Daily_Humidity',\
                            'Daily_Pressure', 'Daily_WindDir', 'Daily_WindSpeed', 'Daily_DNI',\
                            'Daily_DHI', 'Daily_radiation']]

    # produces a 70%, 15%, 15% split for training, validation and test sets
    train_data, validation_data, test_data = np.split(cleaned_df.sample(frac = 1), [int(.7 * len(cleaned_df)), int(.85 * len(cleaned_df))])

    # convert dataframes to .csv and save locally
    train_data.to_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/train.csv', header = True, index = False)
    validation_data.to_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/validation.csv', header = True, index = False)
    test_data.to_csv('~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned/test.csv', header = True, index = False)

