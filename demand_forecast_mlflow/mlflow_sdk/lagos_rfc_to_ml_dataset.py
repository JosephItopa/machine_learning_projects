import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta

credentials = service_account.Credentials.from_service_account_file('./mlflow_sdk/deep-contact-credentials.json')

project_id = 'deep-contact-361614'

def data_extract():
    client = bigquery.Client(credentials= credentials, project=project_id)
    query_job = client.query("""SELECT
                               invoice_number,
                               DATE(invoice_date) AS invoice_date,
                               company_name,
                               company_id,
                               LOWER(company_category) AS company_category,
                               company_address,
                               product_id,
                               product_name,
                               product_category,
                               sub_category,
                               LOWER(uom_) AS uom,
                               LOWER(city) AS city,
                               quantity,
                               vendease_price,
                               total_amount
                            FROM
                                `deep-contact-361614.eproc_processed_data_schema_public.vw_base_product_delivered_item`
                            WHERE
                              DATE(invoice_date) > CURRENT_DATE - 180
                            AND quantity > 0
                            AND product_category NOT IN('meat',
                                    'seafood',
                                    'fruits and vegetables',
                                    'stationery',
                                    'electronics')
                            AND city = 'LAGOS';
                            """) # 
    results = query_job.result()
    tmp = []
    for row in results:
        df = pd.DataFrame([list(row)], columns=['invoice_number', 'invoice_date', 'company_name', 'company_id', 'company_category','company_address','product_id',\
                                                'product_name','product_category','sub_category', 'uom', 'city', 'quantity','vendease_price','total_amount'])
        tmp.append(df)
                    
    df = pd.concat(tmp)
    print(df.head())
    return df #

def lagos_rfc():
    df_22, mainland_lga, island_lga = lagos_rfc_preprocess()
    # RECENCY
    # Evaluation of customer recency
    df_recency = df_22.copy()
    df_recency = df_recency.groupby(by="product_uom", as_index=False)["invoice_date"].max()
    df_recency.columns = ["product_uom", "max_date"]
    
    d = datetime.today() - timedelta(days=90)
    # And an interval of 90 for evaluating recency
    reference_date = pd.to_datetime(str(d.date()))
    df_recency["recency"] = df_recency["max_date"].apply(lambda row: (reference_date - row).days/30)
    df_recency.drop("max_date", inplace=True, axis = 1)
    
    # FREQUENCY
    # Evaluation of number of trips accumulated by customer in 2019
    df_frequency = df_22.copy()
    df_frequency = df_frequency.groupby(by = 'product_uom', as_index = False)['invoice_number'].nunique()
    df_frequency.columns = ['product_uom','frequency']

    # CUSTOMER
    # Evaluation of customer monetary value by aggregating total sum in 2019 per customer
    df_customer = df_22.copy()
    df_customer = df_customer.groupby(by = 'product_uom', as_index = False)['company_name'].nunique()
    df_customer.columns = ['product_uom','customers']
    
    # Merge recency and frequency
    rf_data = df_recency.merge(df_frequency, on = 'product_uom')
    
    # Merge recency, frequency, and customer
    r_f_m_data = rf_data.merge(df_customer, on = 'product_uom')
    
    ## Based on the RFM Evaluation above, Assign R-F-M Score
    r_f_m_data['frequency'] = np.log(r_f_m_data['frequency'])
    r_f_m_data['customers'] = np.log(r_f_m_data['customers'])
    
        # Recency
    def R_Score(x):
        if x['recency'] <= -1.0:
            recency = 3
        elif x['recency'] > -1.0 and x['recency'] <= 0.0:
            recency = 2
        else:
            recency = 1
        return recency

    r_f_m_data['R'] = r_f_m_data.apply(R_Score, axis = 1)
    
        # Frequency
    def F_Score(x):
        if x['frequency'] >= 1.5 and x['frequency'] < 3.0:
            freqency = 2
        elif x['frequency'] >= 3.0:
            freqency = 3
        else:
            freqency = 1
        return freqency

    r_f_m_data['F'] = r_f_m_data.apply(F_Score,axis = 1)
    
        # customer
    def C_Score(x):
        if x['customers'] >= 2.0 and x['customers'] < 3.0:
            count = 2
        elif x['customers'] >= 3.0:
            count = 3
        else:
            count = 1
        return count

    r_f_m_data['C'] = r_f_m_data.apply(C_Score, axis = 1)
    
    def RFC_Score(x):
        return str(x['R']) + str(x['F']) + str(x['C'])

    r_f_m_data['RFC_Score'] = r_f_m_data.apply(RFC_Score,axis=1)
    
    q4result = r_f_m_data[(r_f_m_data["RFC_Score"] == '322') | (r_f_m_data["RFC_Score"] == '333') | (r_f_m_data["RFC_Score"] == '233') |\
                          (r_f_m_data["RFC_Score"] == '232') | (r_f_m_data["RFC_Score"] == '332') | (r_f_m_data["RFC_Score"] == '321') |\
                          (r_f_m_data["RFC_Score"] == '323') | (r_f_m_data["RFC_Score"] == '331')]
    
    no_of_products = q4result.product_uom.nunique()
    
    qproduct_name = list(q4result.product_uom.unique())
    qproducts = df_22.loc[df_22['product_uom'].isin(qproduct_name)]

    df_22i = qproducts[qproducts['lga'].isin(island_lga)]
    df_22m = qproducts[qproducts['lga'].isin(mainland_lga)]
    
    df_22i['area'] = 'Island'
    df_22m['area'] = 'Mainland'
    dfr = pd.concat([df_22i, df_22m])
    
    dff = dfr[(dfr.city=="lagos") & ~(dfr.uom == 'each')]
    
    dff['week_end'] = dff['invoice_date'].dt.to_period('W').dt.end_time
    dff['week_end'] = dff['week_end'].dt.date
    dff['week_start'] = dff['invoice_date'].dt.to_period('W').dt.start_time
    
    dfml = dff.groupby(['week_end','product_name','company_category','product_category','sub_category','uom','area','vendease_price'])\
              .agg({'quantity':'sum'})\
              .reset_index()
    dfm = dfml[(dfml['quantity'] <= 50) & (~(dfml['company_category'] == 'reseller') & ~(dfml['company_category'] == 'others'))]
    generate_training_dataset(q4result)
    ml_pred_dataset(q4result)
    # ------------------------------------------------------
    
    
def lagos_rfc_preprocess():
    df = data_extract()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    df = df[['invoice_number', 'invoice_date', 'company_name', 'company_category','company_address',\
            'product_name', 'product_category', 'sub_category', 'city', 'quantity', 'vendease_price', 'uom',\
            'total_amount']]
    df['uom'].fillna('x', inplace = True)
    df['uom'] = df['uom'].replace({'Cartom':'Carton', 'Tones':'Tonne', 'Liters':'Litres', 'Caton':'Carton', 'Pack':'Packs', 'Kilogram':'Kg'})
    
    df['product_name'] = df['product_name'].astype('str')
    df['product_name'] = df['product_name'].apply(str.lower)
    df['company_name'] = df['company_name'].astype('str')
    df['company_name'] = df['company_name'].apply(str.lower)
    df['uom'] = df['uom'].astype('str')
    df['uom'] = df['uom'].apply(str.lower)
    df['product_category'] = df['product_category'].astype('str')
    df['product_category'] = df['product_category'].apply(str.lower)
    df['sub_category'] = df['sub_category'].astype('str')
    df['sub_category'] = df['sub_category'].apply(str.lower)
    df['company_category'] = df['company_category'].astype('str')
    df['company_category'] = df['company_category'].apply(str.lower)
    df["product_uom"] = df["product_name"]+"_"+df["uom"]
    
    dflag=df[(df.uom != 'each')] #(df.city=="lagos") &

    # copy the dataframe and run the feature engineering function over the data
    dfnl = dflag.copy()
    df_22 = date_features(dfnl)
    
    lgaloc = pd.read_csv("./data/dict/lag_new_location.csv")
    lgaloc['company_address'] = lgaloc['company_address'].astype('str')
    lgaloc['company_address'] = lgaloc['company_address'].apply(str.lower)
    lgaloc['local_gov'] = lgaloc['local_gov'].astype('str')
    lgaloc['local_gov'] = lgaloc['local_gov'].apply(str.lower)
    mapkeyloc = lgaloc[['company_address','local_gov']]
    mapkeyloc.dropna(inplace = True)

    loc_cat = dict(zip(mapkeyloc["company_address"], mapkeyloc["local_gov"]))
    df_22["lga"] = df_22["company_address"].map(loc_cat)
    df_22['lga'] = df_22['lga'].replace({'Obafemi Owode':'Oshodi/Isolo', 'Greater London':'Lagos Island', 'Ilaje':'Lagos Island',\
                                        'Ado Odo/Ota':'Lagos Mainland', 'Egbado North':'Lagos Mainland',\
                                        'Denver County':'Lagos Island', 'Main Land':'Lagos Mainland', 'Mainland':'Lagos Mainland', 'Agege':'Ikeja'})
    
    others = {'orchid':'Eti Osa', 'lekki':'Eti Osa', '20 Admiralty way':'Eti Osa', 'Plot 1A, Block 143 Edward Hotonu Street.':'Eti Osa',\
              '1221 Ahmadu Bello, Vi':'Eti Osa', 'Shop 13 dominion plaza , redemption camp':'Main Land', 'Block C shop 14 canaanland market Redemption camp':'Main Land',\
              'Food share group':'Mainland', 'Block C shop 14 canaanland market Redemption camp':'Main Land', 'Shop 3 ogba garage':'Oshodi/Isolo',\
              'Shop 28 Powa shop Ogba':'Ikeja', 'Shop C37, sabo ultramodern market':'Ikorodu', 'Shop 7 and 8 Mowe Ultramodern Market Mowe':'Oshodi/Isolo',\
              '6,Alashe Road Ajah model market':'Eti Osa', 'D14/C14 Anola oniru modern shopping complex new market':'Eti Osa',\
              'E6 Ajah ultramodern market Ajah':'Eti Osa','3, Animasahun Sunday off Nusurat Lasisi Street Isoloeet,':'Oshodi/Isolo','864A Bishap Ayode Cole':'Eti Osa',\
              'Shop F137-F142,ikota shopping center':'Eti Osa', '1b,Adejumo fajuyita':'Eti Osa', 'No 6, Alhaji Rahmon Saba Close Onibudo Akute':'Ikeja',\
              '9 maitama sule':'Eti Osa','C16 Oniru market':'Eti Osa', '5 Samuel adedoyin, Victoria island, opposite zenith Bank car bank':'Eti Osa',\
              'No A22 new market Ajah':'Eti Osa', 'Shop E18 Oniru Market':'Eti Osa'}
    loc_cat.update(others)
    
    df_22['lga'].fillna('Lagos Mainland', inplace = True)
    mainland_lga = ['Alimosho', 'Ajeromi/Ifelodun', 'Kosofe', 'Mushin', 'Oshodi/Isolo', 'Ikorodu',\
                  'Agege', "Ifako/Ijaiye",'Ikeja','Ojo', 'Shomolu', 'Amuwo-Odofin', 'Badagry', 'Lagos Mainland']
    island_lga = ['Lagos Island', 'Eti Osa', 'Apapa', 'Surulere', 'Epe', 'Ibeju Lekki']
    return df_22, mainland_lga, island_lga

def date_features(df):
    # Date Features
    df['date'] = pd.to_datetime(df['invoice_date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['weekofyear'] = df.date.dt.weekofyear
    df['quarter'] = df.date.dt.quarter
    return df

def generate_training_dataset(q4result):
    df_22, mainland_lga, island_lga = lagos_rfc_preprocess()
    qproduct_name = list(q4result.product_uom.unique())
    qproducts = df_22.loc[df_22['product_uom'].isin(qproduct_name)]
    
    df_22i = qproducts[qproducts['lga'].isin(island_lga)]
    df_22m = qproducts[qproducts['lga'].isin(mainland_lga)]
    
    df_22i['area'] = 'Island'
    df_22m['area'] = 'Mainland'
    dfr = pd.concat([df_22i, df_22m])
    
    dff = dfr[(dfr.city=="lagos") & ~(dfr.uom == 'each')]
    dff['week_end'] = dff['invoice_date'].dt.to_period('W').dt.end_time
    dff['week_end'] = dff['week_end'].dt.date
    dff['week_start'] = dff['invoice_date'].dt.to_period('W').dt.start_time
    dfml = dff.groupby(['week_end','product_name','company_category','product_category','sub_category','uom','area','vendease_price']).agg({'quantity':'sum'}).reset_index()
    dfm = dfml[(dfml['quantity'] <= 50) & (~(dfml['company_category'] == 'reseller') & ~(dfml['company_category'] == 'others'))]
    dfm.to_csv("./data/processed/lag_ml_demand_dataset.csv", index=False)

def ml_pred_dataset(q4result):
    df_22, mainland_lga, island_lga = lagos_rfc_preprocess()
    qproduct_name = list(q4result.product_uom.unique())
    qproducts = df_22.loc[df_22['product_uom'].isin(qproduct_name)]
    
    df_22i = qproducts[qproducts['lga'].isin(island_lga)]
    df_22m = qproducts[qproducts['lga'].isin(mainland_lga)]
    
    df_22i['area'] = 'Island'
    df_22m['area'] = 'Mainland'
    
    sumqtypredi = df_22i.groupby('product_uom', as_index = False)['quantity']\
                    .sum()\
                    .rename(columns = {'product_uom':'product_uom','quantity':'quantity'})
    sumqtypredm = df_22m.groupby('product_uom', as_index = False)['quantity']\
                    .sum()\
                    .rename(columns = {'product_uom':'product_uom','quantity':'quantity'})
    
    sumqtypredi['area'] = 'Island'
    sumqtypredm['area'] = 'Mainland'
    sumqtypred = pd.concat([sumqtypredi, sumqtypredm])
    prod_uom_comcat = df_22[['product_uom', 'company_category']]
    prod_uom_procat = df_22[['product_uom', 'product_category']]
    prod_uom_pricat = df_22[['product_uom', 'vendease_price']]
    prod_uom_comcat.to_csv('./data/dict/prod_uom_comcat_lag.csv', index=False)
    prod_uom_procat.to_csv('./data/dict/prod_uom_procat_lag.csv', index=False)
    prod_uom_pricat.to_csv('./data/dict/prod_uom_pricat_lag.csv', index=False)
    comdict = dict(zip(prod_uom_comcat['product_uom'], prod_uom_comcat['company_category']))
    prodict = dict(zip(prod_uom_procat['product_uom'], prod_uom_procat['product_category']))
    pridict = dict(zip(prod_uom_pricat['product_uom'], prod_uom_pricat['vendease_price']))
    sumqtypred['product_category'] = sumqtypred['product_uom'].map(prodict)
    sumqtypred['company_category'] = sumqtypred['product_uom'].map(comdict)
    sumqtypred['price'] = sumqtypred['product_uom'].map(pridict)
    
    sumqtypred['date'] = np.random.choice(pd.date_range(str(datetime.today().date()), str((datetime.today() + timedelta(days=30)).date())), sumqtypred.shape[0])
    
    realdf = sumqtypred[['date', 'product_uom', 'area','product_category', 'company_category', 'price']]
    realdf = realdf[~(realdf.company_category=='commodity trader') & ~(realdf.company_category=='reseller')]
    realdf.to_csv("./data/processed/lag_pred_for_qty.csv", index=False)