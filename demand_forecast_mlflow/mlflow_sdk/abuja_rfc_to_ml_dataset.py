import json
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta

credentials = service_account.Credentials.from_service_account_file('./mlflow_sdk/deep-contact-credentials.json')

project_id = 'deep-contact-361614'
        
def data_extract():
    print("extracting data...")
    client = bigquery.Client(credentials= credentials,project=project_id)
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
                                AND city = 'ABUJA';
                            """)
    results = query_job.result()
    tmp = []
    for row in results:
        df = pd.DataFrame([list(row)], columns=['invoice_number','invoice_date','company_name','company_id','company_category','company_address',
                                            'product_id','product_name','product_category','sub_category','uom','city','quantity',
                                            'vendease_price','total_amount'])
        tmp.append(df)
    df=pd.concat(tmp)
    print(df.head())
    return df

def abj_rfc_process():
    df = data_extract()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df = df[['invoice_number', 'invoice_date', 'company_name', 'company_category',\
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
    
    df_abj = df[(df.uom != 'each')] # (df.city == "ABUJA")
    
    # copy the dataframe and run the feature engineering function over the data
    dfnl = df_abj.copy()
    dfnabj = date_features(dfnl)
    print(dfnabj.head())
    print("abuja_rfc processing")
    return dfnabj
    
    # DATES FEATURES
def date_features(df):
    # Date Features
    df['date'] = pd.to_datetime(df['invoice_date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['weekofyear'] = df.date.dt.weekofyear
    df['quarter'] = df.date.dt.quarter
    print("success for date feature extraction")
    return df

def abuja_rfc():

    df_22 = abj_rfc_process()
    # Evaluation of customer recency
    df_recency = df_22.copy()
    df_recency = df_recency.groupby(by="product_uom", as_index=False)["invoice_date"].max()
    df_recency.columns = ["product_uom", "max_date"]
    
    d = datetime.today() - timedelta(days=90)
    # And an interval of 90 for evaluating recency
    reference_date = pd.to_datetime(str(d.date()))
    print("reference_date",reference_date)
    df_recency["recency"] = df_recency["max_date"].apply(lambda row: (reference_date - row).days/30)
    df_recency.drop("max_date", inplace=True, axis = 1)

    # Evaluation of Frequency
    df_frequency = df_22.copy()
    df_frequency = df_frequency.groupby(by = 'product_uom', as_index = False)['invoice_number'].nunique()
    df_frequency.columns = ['product_uom','frequency']
    
    # Evaluation of customer
    df_customer = df_22.copy()
    df_customer = df_customer.groupby(by = 'product_uom', as_index = False)['company_name'].nunique()
    df_customer.columns =  ['product_uom','customers']
    
    # Merge recency and frequency
    rf_data = df_recency.merge(df_frequency, on = 'product_uom')
    # Merge recency, frequency, and customer
    r_f_m_data = rf_data.merge(df_customer, on = 'product_uom')
    
    ### Based on the RFM Evaluation above, Assign R-F-M Score
    r_f_m_data['frequency'] = np.log(r_f_m_data['frequency'])
    r_f_m_data['customers'] = np.log(r_f_m_data['customers'])
    
    # Recency
    def R_Score(x):
        if x['recency'] < -1.0:
            recency = 3
        elif x['recency'] >= -1.0 and x['recency'] <= 0.0:
            recency = 2
        else:
            recency = 1
        return recency

    r_f_m_data['R'] = r_f_m_data.apply(R_Score, axis = 1)
    
    # Frequency
    def F_Score(x):
        if x['frequency'] > 1.5 and x['frequency'] < 3.0:
            freqency = 2
        elif x['frequency'] > 3.0:
            freqency = 3
        else:
            freqency = 1
        return freqency

    r_f_m_data['F'] = r_f_m_data.apply(F_Score,axis = 1)
    
    # customer
    def C_Score(x):
        if x['customers'] > 1 and x['customers'] < 2:
            count = 2
        elif x['customers'] >= 2:
            count = 3
        else:
            count = 1
        return count

    r_f_m_data['C'] = r_f_m_data.apply(C_Score, axis = 1)
    
    def RFC_Score(x):
        return str(x['R']) + str(x['F']) + str(x['C'])

    r_f_m_data['RFC_Score'] = r_f_m_data.apply(RFC_Score, axis = 1)
    print("r_f_m_data", r_f_m_data)
    q4result = r_f_m_data[(r_f_m_data["RFC_Score"]=='321') | (r_f_m_data["RFC_Score"]=='332') | (r_f_m_data["RFC_Score"]=='233') |\
                          (r_f_m_data["RFC_Score"]=='333') | (r_f_m_data["RFC_Score"]=='323') | (r_f_m_data["RFC_Score"]=='331') |\
                          (r_f_m_data["RFC_Score"]=='322') | (r_f_m_data["RFC_Score"]=='232')]
    
    #print("q4result filter", r_f_m_data["RFC_Score"].value_counts())
    prod_uom_comcat = df_22[['product_uom', 'company_category']]
    #print(prod_uom_comcat.head())
    prod_uom_procat = df_22[['product_uom', 'product_category']]
    #print(prod_uom_procat.head())
    prod_uom_pricat = df_22[['product_uom', 'vendease_price']]
    #print(prod_uom_pricat.head())
    prod_uom_comcat.to_csv('./data/dict/prod_uom_comcat.csv', index = False)
    prod_uom_procat.to_csv('./data/dict/prod_uom_procat.csv', index = False)
    prod_uom_pricat.to_csv('./data/dict/prod_uom_pricat.csv', index = False)
    comdict = dict(zip(prod_uom_comcat['product_uom'], prod_uom_comcat['company_category']))
    prodict = dict(zip(prod_uom_procat['product_uom'], prod_uom_procat['product_category']))
    pridict = dict(zip(prod_uom_pricat['product_uom'], prod_uom_pricat['vendease_price']))
    q4result['product_category'] = q4result['product_uom'].map(prodict)
    q4result['company_category'] = q4result['product_uom'].map(comdict)
    q4result['vendease_price'] = q4result['product_uom'].map(pridict)
    #print("q4result after dict", q4result.head())
    q4result['date'] = np.random.choice(pd.date_range(str(datetime.today().date()), str((datetime.today() + timedelta(days=30)).date())), q4result.shape[0])
    #print("q4result",q4result.head())
    realdf = q4result[['date', 'product_uom', 'product_category', 'company_category', 'vendease_price']]
    realdf.to_csv("./data/processed/abuja_pred_for_qty.csv",index=False)
    generate_training_data(q4result)
    print("abuja_rfc finished processing")

def generate_training_data(q4result):
    df =  abj_rfc_process()
    dff = df[(df.uom != 'each')] #(df.city=="") & 
    q4product_name = list(q4result.product_uom.unique())
    q4products = dff.loc[dff['product_uom'].isin(q4product_name)] #
    
    q4products['week_end'] = q4products['invoice_date'].dt.to_period('W').dt.end_time
    q4products['week_end'] = q4products['week_end'].dt.date
    q4products['week_start'] = q4products['invoice_date'].dt.to_period('W').dt.start_time
    
    dfml = q4products.groupby(['week_end','product_name','company_category','product_category','sub_category','uom','vendease_price'])\
                     .agg({'quantity':'sum'}).reset_index()
    dfm = dfml[(dfml['quantity'] <= 50)]
    #print("dfm", dfm.head())
    dfm.to_csv("./data/processed/ml_demand_dataset.csv", index=False)
    print("generated_data finished processing")

#abj_rfc_process()
#abuja_rfc()