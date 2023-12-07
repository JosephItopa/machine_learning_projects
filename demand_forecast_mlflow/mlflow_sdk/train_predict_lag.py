import json
import joblib
import random
import numpy as np
import pandas as pd
from .lagos_rfc_to_ml_dataset import lagos_rfc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_():
    lagos_rfc()
    df=pd.read_csv("./data/processed/lag_ml_demand_dataset.csv")

    df = df.drop(['sub_category'], axis=1)
    df['sku']=df['product_name']+"_"+df['uom']

    dock=list(df['sku'].unique())
    sku_dock=dict(enumerate(dock))
    
    skud = pd.DataFrame(list(sku_dock.items()), columns = ['labels', 'product_name'])
    skud = skud[['product_name', 'labels']]
    skud.to_csv('lag_prod_dict.csv', index=False)
    skudict=dict(zip(skud["product_name"], skud["labels"]))
    df['sku_no'] = df['sku'].map(skudict)
    df = df.drop(['product_name', 'uom', 'sku'], axis=1)
    
    avg=df.groupby("week_end")["quantity","vendease_price"].mean().reset_index()
    
    dfo = check_outliers(df, ['vendease_price','quantity'], 5)
    df = dfo[~dfo.outlier_vendease_price]
    df = dfo[~dfo.outlier_quantity]
    df['week_end'] = pd.to_datetime(df['week_end'])
    df['trend']=df['week_end'].dt.year - 2022
    df['month']=df['week_end'].dt.month
    
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
    df = df[~(df["company_category"] == 'mart')]#
    df["area"].replace({"Mainland":1, "Island":0}, inplace=True)
    df = pd.get_dummies(data=df, columns=['company_category', 'product_category'], drop_first = True)
    df=df.sort_values(by=['sku_no','week_end'])
    
    df.to_csv('./data/processed/lag_sku_data_processed.csv', index=False)
    
    return df
    
def model_training_():
    params=[]
    maximum_score=0
    max_features_ = list(range(2,45))
    max_depth_ = list(range(2,10))
    learning_rate_ = [0.01, 0.05, 0.1, 0.25, 0.5]
    
    sales = preprocess_()#pd.read_csv("lag_sku_june_data_processed_v1.csv")
    sales['week_end'] = pd.to_datetime(sales['week_end'])
    sales['weekofyear'] = sales.week_end.dt.weekofyear
    X = sales.drop(['week_end','quantity','month','trend'], axis=1)
    y = sales['quantity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
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
        GB_cen=GradientBoostingRegressor(n_estimators=200, max_features=mf,max_depth=md,learning_rate=lr,random_state=0).ﬁt(X_train,y_train)
        score=r2_score(y_test, GB_cen.predict(X_test))
        print('R2:',score)
        #compare performances on validation data
        if score > maximum_score:
            params = [mf,md,lr]
            maximum_score = score
    
    ## Test on fresh data
    mf,md,lr = params

    print('\nBest Model:')
    print('Parameters:',params)
    print('Validation R2:',maximum_score)
    
    return mf, md, lr, X_train, X_test, y_train, y_test

def model_predict():
    df = pd.read_csv("./data/dict/lag_pred_june_for_qty.csv")
    priceload = df[["product_uom","price"]]
    priceload.to_csv("./data/dict/productprice.csv", index = False)
    
    skud = pd.read_csv('./data/dict/lag_prod_dict.csv')
    skudict = dict(zip(skud["product_name"], skud["labels"]))
    df['sku_no'] = df['product_uom'].map(skudict)
    
    df['date'] = pd.to_datetime(df['date'])
    #df['trend'] = df['date'].dt.year - 2021
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.weekofyear
    
    ## Lag prices
    df["price-1"] = df["price"]
    df["price-2"] = df["price"]
    
    #price
    col = df.pop("price")
    df.insert(3, col.name, col)
    pos_price = df.columns.get_loc('price')
    #p-1
    col = df.pop("price-1")
    df.insert(pos_price+1, col.name, col)
    #p-2
    col = df.pop("price-2")
    df.insert(pos_price+2, col.name, col)
    
    df = pd.get_dummies(data = df, columns = ['company_category','product_category'], drop_first = True)
    df.dropna(inplace=True)
    df["area"].replace({"Mainland":1,"Island":0}, inplace=True)
    
def check_outliers(df,features,k=5):
    data = df.copy()
    for f in features:
        # data['mean+'+str(k)+'*std_'+f] = data.groupby('sku')[f].transform(
        # lambda x: x.mean()+k*x.std()  )
        # data['mean-'+str(k)+'*std_'+f] = data.groupby('sku')[f].transform(
        # lambda x: x.mean()-k*x.std()  )
        data['outlier_'+f] = data.groupby('sku_no')[f].transform(\
                                                                 lambda x: (x > (x.mean()+k*x.std()))  |(x < (x.mean()-k*x.std())))
    return(data)