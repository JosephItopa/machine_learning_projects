import json
import joblib
import random
import numpy as np
import pandas as pd
from .abuja_rfc_to_ml_dataset import abuja_rfc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_():
    abuja_rfc()
    df= pd.read_csv("./data/processed/ml_demand_dataset.csv")
    df['sub_category'].fillna('x', inplace=True)
    df['sku']=df['product_name']+"_"+df['uom']
    dock = list(df['sku'].unique())
    sku_dock = dict(enumerate(dock))
    skud = pd.DataFrame(list(sku_dock.items()), columns = ['labels', 'product_name'])
    skud = skud[['product_name', 'labels']]
    
    skud = pd.DataFrame(list(sku_dock.items()), columns = ['labels', 'product_name'])
    skud = skud[['product_name', 'labels']]
    skud.to_csv('./data/dict/abuja_prod_dict.csv', index=False)
    skudict=dict(zip(skud["product_name"], skud["labels"]))
    
    df['sku_no'] = df['sku'].map(skudict)
    df = df.drop(['product_name', 'uom', 'sku','sub_category'], axis = 1)
    df = df[~(df['product_category'] == 'poultry')]

    dfo = check_outliers(df, ['vendease_price','quantity'], 5)
    
    df = dfo[~dfo.outlier_vendease_price]
    df = dfo[~dfo.outlier_quantity]
    
    ## Lag prices
    df["price-1"] = df.groupby(['sku_no'])['vendease_price'].shift(1)
    df["price-2"] = df.groupby(['sku_no'])['vendease_price'].shift(2)
    df.dropna(subset=['price-1',"price-2"],inplace=True)

    #price
    col = df.pop("vendease_price")
    df.insert(3, col.name, col)
    pos_price = df.columns.get_loc('vendease_price')
    #p-1
    col = df.pop("price-1")
    df.insert(pos_price+1, col.name, col)
    #p-2
    col = df.pop("price-2")
    df.insert(pos_price+2, col.name, col)
    
    df = df[(df['company_category']!="others") & (df['company_category']!="commodity trader") & (df['company_category']!="reseller") &\
        (df['product_category']!="toiletries") & (df['product_category']!="baking goods") & (df['product_category']!="stationery")]
    
    df = pd.get_dummies(data=df, columns=['company_category','product_category'], drop_first = True)
    
    df.to_csv('./data/processed/sku_data_processed.csv', index=False)
    
    return df
    
def check_outliers(df,features,k=5):
    data = df.copy()
    for f in features:
        # data['mean+'+str(k)+'*std_'+f] = data.groupby('sku')[f].transform(
        # lambda x: x.mean()+k*x.std()  )
        # data['mean-'+str(k)+'*std_'+f] = data.groupby('sku')[f].transform(
        # lambda x: x.mean()-k*x.std()  )
        data['outlier_'+f] = data.groupby('sku_no')[f].transform(
                                lambda x: (x > (x.mean()+k*x.std())) | (x < (x.mean()-k*x.std())))
    return(data)
    
    
    
def model_training():
    sales = preprocess_() #pd.read_csv("./data/processed/sku_data_processed.csv")
    sales['week_end'] = pd.to_datetime(sales['week_end'])
    sales['weekofyear'] = sales.week_end.dt.weekofyear
    
    X = sales.drop(['week_end','quantity','price-1','price-2'], axis=1)
    y = sales['quantity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
    
    max_features_ = list(range(2,45))
    max_depth_ = list(range(2,10))
    learning_rate_ = [0.75, 0.1, 0.125, 0.15, 0.175, 0.2] #
    params=[]
    maximum_score=0
    
    #selection of parameters to test
    random.seed(5)
    mf_ = random.choices(max_features_, k=50)
    md_ = random.choices(max_depth_, k=50)
    lr_ = random.choices(learning_rate_, k=50)
    ## Iterations to select best model
    for i in range(50):
        print('Model number:',i+1)
        #selection of parameters to test
        mf = mf_[i]
        md = md_[i]
        lr = lr_[i]
        print('Parameters:',[mf,md,lr])
        #model
        GB_cen=GradientBoostingRegressor(n_estimators=200,max_features=mf,max_depth=md,learning_rate=lr,random_state=0).ﬁt(X_train,y_train)
        score=r2_score(y_test, GB_cen.predict(X_test))
        print('R2:',score)
        #compare performances on validation data
        if score > maximum_score:
            params = [mf,md,lr]
            maximum_score = score
            
    ## filter best parameter and return data and parameter
    mf,md,lr = params
    return mf, md, lr, X_train, X_test, y_train, y_test