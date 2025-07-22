import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import pandas as pd
from simple_salesforce import Salesforce
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --- SETUP AND CONFIGURATION ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# --- SALESFORCE CONNECTION (Global for simplicity) ---
SF_USERNAME = os.getenv("SALESFORCE_USERNAME")
SF_PASSWORD = os.getenv("SALESFORCE_PASSWORD")
SF_SECURITY_TOKEN = os.getenv("SALESFORCE_SECURITY_TOKEN")
SF_DOMAIN = 'login'

sf_connection = None
try:
    sf_connection = Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_SECURITY_TOKEN,
        domain=SF_DOMAIN
    )
    print("✅ Successfully connected to Salesforce!")
except Exception as e:
    print(f"❌ Error connecting to Salesforce: {e}")

# --- HELPER FUNCTION FOR JSON SERIALIZATION ---
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# --- DATA FETCHING AND TRANSFORMATION FUNCTIONS ---

def get_comparison_data(data_function, start_date_str, end_date_str, sales_region, **kwargs):
    """
    A wrapper function to get data for a current period, its preceding period, and the same period last year.
    """
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        delta = end_date - start_date
        
        # Previous period
        prev_end_date = start_date - timedelta(days=1)
        prev_start_date = prev_end_date - delta
        prev_start_date_str = prev_start_date.strftime('%Y-%m-%d')
        prev_end_date_str = prev_end_date.strftime('%Y-%m-%d')

        # Same period last year
        last_year_start_date = start_date - relativedelta(years=1)
        last_year_end_date = end_date - relativedelta(years=1)
        last_year_start_date_str = last_year_start_date.strftime('%Y-%m-%d')
        last_year_end_date_str = last_year_end_date.strftime('%Y-%m-%d')

        current_data = data_function(start_date_str, end_date_str, sales_region, **kwargs)
        previous_data = data_function(prev_start_date_str, prev_end_date_str, sales_region, **kwargs)
        last_year_data = data_function(last_year_start_date_str, last_year_end_date_str, sales_region, **kwargs)
        
        return {'current': current_data, 'previous': previous_data, 'last_year': last_year_data}
        
    except Exception as e:
        print(f"❌ Period comparison calculation error for {data_function.__name__}: {e}")
        return {'current': data_function(start_date_str, end_date_str, sales_region, **kwargs), 'previous': {}, 'last_year': {}}


def get_forecast_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    where_clauses = ["Primary_Product__c != 'Servers'"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT Amount, Sales_Region__c, ForecastCategoryName, StageName FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        query_result = sf_connection.query_all(soql_query)
        df = pd.DataFrame(query_result['records']).drop(columns=['attributes'])
    except Exception as e:
        print(f"❌ Forecast Query Error: {e}")
        return {}
    if df.empty: return {}
    df = df[df['StageName'] != 'Closed Lost']
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    pivot_df = df.groupby(['Sales_Region__c', 'ForecastCategoryName'])['Amount'].sum().unstack(fill_value=0)
    region_summary = df.groupby('Sales_Region__c').agg(Opp_Count=('Amount', 'count'), Total_Amount=('Amount', 'sum'))
    sorted_regions = region_summary.sort_values(by='Total_Amount', ascending=False).index
    pivot_df = pivot_df.reindex(sorted_regions)
    region_summary = region_summary.reindex(sorted_regions)
    forecast_order = ['Pipeline', 'Best Case', 'Commit', 'Closed']
    existing_columns = [col for col in forecast_order if col in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]
    return {'regions': pivot_df.index.tolist(), 'categories': pivot_df.columns.tolist(), 'bar_data': {col: pivot_df[col].tolist() for col in pivot_df.columns}, 'line_data': {'regions': region_summary.index.tolist(), 'counts': region_summary['Opp_Count'].tolist()}}

def get_bookings_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    booking_products = "('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex')"
    where_clauses = ["StageName = 'Closed Won'", f"Primary_Product__c IN {booking_products}"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT CALENDAR_YEAR(CloseDate) c_year, CALENDAR_MONTH(CloseDate) c_month, Sales_Region__c, SUM(Amount) totalAmount FROM Opportunity WHERE {' AND '.join(where_clauses)} GROUP BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate), Sales_Region__c ORDER BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate)"
    try:
        query_result = sf_connection.query_all(soql_query)
        df = pd.DataFrame(query_result['records'])
    except Exception as e:
        print(f"❌ Bookings Query Error: {e}")
        return {}
    if df.empty: return {}
    df['label'] = pd.to_datetime(df['c_year'].astype(str) + '-' + df['c_month'].astype(str) + '-01').dt.strftime('%b %Y')
    pivot_df = df.pivot_table(index='label', columns='Sales_Region__c', values='totalAmount', aggfunc='sum').fillna(0)
    pivot_df.index = pd.to_datetime(pivot_df.index, format='%b %Y')
    pivot_df = pivot_df.sort_index()
    pivot_df.index = pivot_df.index.strftime('%b %Y')
    return {'labels': pivot_df.index.tolist(), 'region_data': {col: pivot_df[col].tolist() for col in pivot_df.columns}, 'regions': pivot_df.columns.tolist()}

def get_goal_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    booking_products = "('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex')"
    shared_where = []
    if start_date: shared_where.append(f"CloseDate >= {start_date}")
    if end_date: shared_where.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        shared_where.append(f"Sales_Region__c IN ('{formatted_regions}')")
    shared_where_string = f"WHERE {' AND '.join(shared_where)}" if shared_where else ""
    numerator_query = f"SELECT SUM(Amount) FROM Opportunity {shared_where_string} {'AND' if shared_where else 'WHERE'} StageName = 'Closed Won' AND Primary_Product__c IN {booking_products}"
    denominator_query = f"SELECT SUM(Amount_Goal__c) FROM Opportunity {shared_where_string} {'AND' if shared_where else 'WHERE'} RecordType.Name = 'Quota' AND StageName = 'Quota'"
    try:
        numerator_result = sf_connection.query(numerator_query)
        denominator_result = sf_connection.query(denominator_query)
        numerator = numerator_result['records'][0]['expr0'] or 0
        denominator = denominator_result['records'][0]['expr0'] or 0
        percentage = (numerator / denominator) * 100 if denominator != 0 else 0
    except Exception as e:
        print(f"❌ Goal Query Error: {e}")
        return {'percentage': 0}
    return {'percentage': round(percentage, 2)}

def get_booked_revenue_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    booking_products = "('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex')"
    where_clauses = ["IsWon = True", f"Primary_Product__c IN {booking_products}"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT SUM(Amount) FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        result = sf_connection.query(soql_query)
        booked_revenue = result['records'][0]['expr0'] or 0
    except Exception as e:
        print(f"❌ Booked Revenue Query Error: {e}")
        return {'booked_revenue': 0}
    return {'booked_revenue': booked_revenue}

def get_new_storage_customers_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    storage_products = "('FreeNAS Cert', 'TrueNAS')"
    where_clauses = ["StageName = 'Closed Won'", f"Primary_Product__c IN {storage_products}", "Type = 'New Business'"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT COUNT(Id) FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        result = sf_connection.query(soql_query)
        customer_count = result['records'][0]['expr0'] or 0
    except Exception as e:
        print(f"❌ New Storage Customers Query Error: {e}")
        return {'customer_count': 0}
    return {'customer_count': customer_count}

def get_repeat_storage_customers_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    storage_products = "('FreeNAS Cert', 'TrueNAS')"
    where_clauses = ["StageName = 'Closed Won'", f"Primary_Product__c IN {storage_products}", "Type = 'Existing Business'"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT COUNT(Id) FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        result = sf_connection.query(soql_query)
        customer_count = result['records'][0]['expr0'] or 0
    except Exception as e:
        print(f"❌ Repeat Storage Customers Query Error: {e}")
        return {'customer_count': 0}
    return {'customer_count': customer_count}

def get_large_deals_data(start_date, end_date, sales_region, amount_threshold):
    if not sf_connection: return {}
    where_clauses = ["IsWon = True", "Primary_Product__c != 'Servers'", f"Amount >= {amount_threshold}"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT COUNT(Id) FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        result = sf_connection.query(soql_query)
        deal_count = result['records'][0]['expr0'] or 0
    except Exception as e:
        print(f"❌ Large Deals Query Error (Amount >= {amount_threshold}): {e}")
        return {'deal_count': 0}
    return {'deal_count': deal_count}

def get_current_open_commit_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    commit_products = "('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex')"
    where_clauses = ["IsClosed = False", "ForecastCategoryName = 'Commit'", f"Primary_Product__c IN {commit_products}"]
    if start_date: where_clauses.append(f"CloseDate >= {start_date}")
    if end_date: where_clauses.append(f"CloseDate <= {end_date}")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT SUM(Amount) FROM Opportunity WHERE {' AND '.join(where_clauses)}"
    try:
        result = sf_connection.query(soql_query)
        commit_amount = result['records'][0]['expr0'] or 0
    except Exception as e:
        print(f"❌ Current Open Commit Query Error: {e}")
        return {'commit_amount': 0}
    return {'commit_amount': commit_amount}

def get_current_forecast_data(start_date, end_date, sales_region):
    booked_data = get_booked_revenue_data(start_date, end_date, sales_region)
    commit_data = get_current_open_commit_data(start_date, end_date, sales_region)
    booked_revenue = booked_data.get('booked_revenue', 0)
    open_commit = commit_data.get('commit_amount', 0)
    current_forecast = booked_revenue + open_commit
    return {'current_forecast': current_forecast}

def get_rep_performance_data(start_date, end_date):
    if not sf_connection: return {}
    owner_exclusions = "('Casey Abbott', 'Grace de Leon', 'Aundria Giardina')"
    booking_products = ('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex', None, '')
    shared_where = []
    if start_date: shared_where.append(f"CloseDate >= {start_date}")
    if end_date: shared_where.append(f"CloseDate <= {end_date}")
    shared_where_string = ' AND '.join(shared_where)
    soql_query = f"SELECT Owner.Name, Amount, Amount_Goal__c, StageName, Primary_Product__c, Account.Name, Quote_Number__c, Quota_Period__c, CloseDate FROM Opportunity WHERE {shared_where_string} {'AND' if shared_where else ''} Owner.Name NOT IN {owner_exclusions}"
    try:
        query_result = sf_connection.query_all(soql_query)
        if not query_result['records']: return {}
        records = [{'ownerName': rec['Owner']['Name'], 'Amount': rec['Amount'], 'Amount_Goal__c': rec['Amount_Goal__c'], 'StageName': rec['StageName'], 'Primary_Product__c': rec['Primary_Product__c'], 'AccountName': rec['Account']['Name'] if rec['Account'] else '', 'Quote_Number__c': rec['Quote_Number__c'], 'Quota_Period__c': rec['Quota_Period__c']} for rec in query_result['records']]
        df = pd.DataFrame(records)
        quota_df = df[(df['StageName'] == 'Quota') & (df['Quota_Period__c'] == 'Monthly')]
        quota_agg = quota_df.groupby('ownerName')['Amount_Goal__c'].sum().reset_index().rename(columns={'Amount_Goal__c': 'totalQuota'})
        bookings_df = df[(df['StageName'] == 'Closed Won') & (~df['AccountName'].str.contains('ixsystems|iX Amazon', case=False, na=False)) & (~df['Quote_Number__c'].str.contains('configurated', case=False, na=False)) & (df['Primary_Product__c'].isin(booking_products))]
        bookings_agg = bookings_df.groupby('ownerName')['Amount'].sum().reset_index().rename(columns={'Amount': 'totalBookings'})
        if quota_agg.empty: return {}
        merged_df = pd.merge(quota_agg, bookings_agg, on='ownerName', how='left').fillna(0)
        merged_df = merged_df[merged_df['totalQuota'] > 0]
        merged_df['percent_to_quota'] = merged_df.apply(lambda row: (row['totalBookings'] / row['totalQuota'] * 100), axis=1)
        merged_df = merged_df.sort_values(by='percent_to_quota', ascending=False)
    except Exception as e:
        print(f"❌ Rep Performance Query Error: {e}")
        return {}
    return {'reps': merged_df['ownerName'].tolist(), 'percentages': merged_df['percent_to_quota'].tolist()}

def get_sdr_activity_data(start_date, end_date, activity_type):
    if not sf_connection: return {}
    sdr_names = "('Sheen Trisal', 'Brandon Mazikowski', 'Hayden Barcelos', 'Jaylen Macias-Matsuura')"
    where_clauses = [f"Owner.Name IN {sdr_names}", f"Type = '{activity_type}'", f"ActivityDate <= {datetime.now().strftime('%Y-%m-%d')}"]
    if activity_type == 'Email': where_clauses.append("Subject LIKE '%Gong%'")
    if activity_type == 'Meeting': where_clauses.append("Owner.Name != 'IT Dept'")
    effective_start_date = start_date if start_date else f"{datetime.now().year}-01-01"
    where_clauses.append(f"ActivityDate >= {effective_start_date}")
    if end_date: where_clauses.append(f"ActivityDate <= {end_date}")
    soql_query = f"SELECT Owner.Name, ActivityDate FROM Task WHERE {' AND '.join(where_clauses)}"
    try:
        query_result = sf_connection.query_all(soql_query)
        records = [{'ownerName': rec['Owner']['Name'], 'ActivityDate': rec['ActivityDate']} for rec in query_result['records']]
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"❌ SDR {activity_type} Query Error: {e}")
        return {}
    if df.empty: return {}
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    df['label'] = df['ActivityDate'].dt.strftime('%Y-%m')
    grouped_df = df.groupby(['label', 'ownerName']).size().reset_index(name='count')
    pivot_df = grouped_df.pivot_table(index='label', columns='ownerName', values='count', aggfunc='sum').fillna(0)
    pivot_df.index = pd.to_datetime(pivot_df.index, format='%b %Y')
    pivot_df.index = pivot_df.index.strftime('%b %Y')
    return {'labels': pivot_df.index.tolist(), 'sdr_data': {col: pivot_df[col].tolist() for col in pivot_df.columns}, 'sdr_names': pivot_df.columns.tolist()}

def get_sdr_generated_opps_chart_data(start_date, end_date, sales_region):
    if not sf_connection: return {}
    sdr_names = "('Sheen Trisal','Brandon Mazikowski','Hayden Barcelos','Jaylen Macias-Matsuura')"
    stages = "('Qualification','Proposal','Negotiation','Expected','Closed Won','Cancelled','Closed Lost')"
    where_clauses = [f"SDR_on_Opportunity__r.Name IN {sdr_names}", "Lead_Pipe__c = 'Outbound'", f"StageName IN {stages}"]
    if start_date: where_clauses.append(f"CreatedDate >= {start_date}T00:00:00Z")
    if end_date: where_clauses.append(f"CreatedDate <= {end_date}T23:59:59Z")
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    soql_query = f"SELECT SDR_on_Opportunity__r.Name sdrName, CALENDAR_YEAR(CreatedDate) c_year, CALENDAR_MONTH(CreatedDate) c_month, COUNT(Id) oppCount FROM Opportunity WHERE {' AND '.join(where_clauses)} GROUP BY SDR_on_Opportunity__r.Name, CALENDAR_YEAR(CreatedDate), CALENDAR_MONTH(CreatedDate) ORDER BY CALENDAR_YEAR(CreatedDate), CALENDAR_MONTH(CreatedDate)"
    try:
        query_result = sf_connection.query_all(soql_query)
        records = [{'sdrName': rec['sdrName'], 'c_year': rec['c_year'], 'c_month': rec['c_month'], 'oppCount': rec['oppCount']} for rec in query_result['records']]
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"❌ SDR Generated Opps Chart Query Error: {e}")
        return {}
    if df.empty: return {}
    df['label'] = pd.to_datetime(df['c_year'].astype(str) + '-' + df['c_month'].astype(str) + '-01').dt.strftime('%b %Y')
    pivot_df = df.pivot_table(index='label', columns='sdrName', values='oppCount', aggfunc='sum').fillna(0)
    pivot_df.index = pd.to_datetime(pivot_df.index, format='%b %Y')
    pivot_df = pivot_df.sort_index()
    pivot_df.index = pivot_df.index.strftime('%b %Y')
    return {'labels': pivot_df.index.tolist(), 'sdr_data': {col: pivot_df[col].tolist() for col in pivot_df.columns}, 'sdr_names': pivot_df.columns.tolist()}

def get_ml_predictions(months_lookback, opportunity_type):
    if not sf_connection: return {"error": "Salesforce connection not available."}
    
    TARGET_OBJECT_ML = 'Opportunity'
    PRIMARY_PRODUCT_FILTER = 'TrueNAS'
    TARGET_VARIABLE = 'IsWon'
    
    features_to_select = ['Amount', 'LeadSource', 'Type', 'Sales_Program__c', 'Sales_Region__c', 'CreatedDate', 'CloseDate', 'IsWon', 'IsClosed', 'StageName']
    
    where_clauses_ml = ["IsClosed = TRUE", f"Primary_Product__c = '{PRIMARY_PRODUCT_FILTER}'"]
    
    if months_lookback:
        try:
            lookback_date_str = (datetime.now() - relativedelta(months=int(months_lookback))).strftime('%Y-%m-%d')
            where_clauses_ml.append(f"CloseDate >= {lookback_date_str}")
        except ValueError:
            return {"error": "Invalid months lookback value."}

    if opportunity_type:
        if opportunity_type in ['New Business', 'Existing Business']:
            where_clauses_ml.append(f"Type = '{opportunity_type}'")
        elif opportunity_type == 'Support Renewals':
            where_clauses_ml.append("Sales_Program__c = 'SUP/WRNY Renewal'")
        elif opportunity_type == 'System Refresh':
            where_clauses_ml.append("Sales_Program__c = 'System Refresh'")

    soql_ml_query = f"SELECT {', '.join(features_to_select)} FROM {TARGET_OBJECT_ML} WHERE {' AND '.join(where_clauses_ml)}"
    
    try:
        print(f"⏳ Fetching data for ML Model with query: {soql_ml_query}")
        ml_records = sf_connection.query_all_iter(soql_ml_query)
        df_ml = pd.DataFrame(list(ml_records))
        if 'attributes' in df_ml.columns:
            df_ml = df_ml.drop(columns=['attributes'])
        if df_ml.empty or len(df_ml) < 20:
            return {"error": "Not enough opportunity data found for the selected filters to train a reliable model."}
        print(f"✅ Successfully fetched {len(df_ml)} opportunities for ML.")
    except Exception as e:
        return {"error": f"ML Data Fetch Error: {e}"}

    df_ml['IsWon'] = df_ml['IsWon'].astype(bool)
    df_ml['CloseDate'] = pd.to_datetime(df_ml['CloseDate'], errors='coerce')
    df_ml['CreatedDate'] = pd.to_datetime(df_ml['CreatedDate'], errors='coerce').dt.tz_localize(None)
    df_ml['OpportunityAge'] = (df_ml['CloseDate'] - df_ml['CreatedDate']).dt.days
    
    X = df_ml[['Amount', 'LeadSource', 'Sales_Region__c', 'OpportunityAge']]
    y = df_ml['IsWon']
    
    numerical_features = ['Amount', 'OpportunityAge']
    categorical_features = ['LeadSource', 'Sales_Region__c']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42, stratify=y)
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "The filtered data contains only one outcome (all Won or all Lost), so a predictive model cannot be trained."}

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("⏳ Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("✅ Model trained.")
    
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(30)
    
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    final_results = {
        "test_accuracy": test_accuracy,
        "feature_importances": feature_importance_df.to_dict(orient='records'),
        "classification_report": report_dict
    }

    return convert_numpy_types(final_results)


def get_sales_regions():
    if not sf_connection: return []
    try:
        records = sf_connection.query("SELECT Sales_Region__c FROM Opportunity GROUP BY Sales_Region__c", include_deleted=False)
        regions = [record['Sales_Region__c'] for record in records['records'] if record['Sales_Region__c'] is not None]
        return sorted(regions)
    except Exception as e:
        print(f"❌ Sales Regions Query Error: {e}")
        return []

# --- FLASK ROUTES ---
@app.route('/api/data/forecast')
def forecast_data_endpoint():
    return jsonify(get_forecast_data(request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/bookings')
def bookings_data_endpoint():
    return jsonify(get_bookings_data(request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/goal')
def goal_data_endpoint():
    return jsonify(get_comparison_data(get_goal_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/booked-revenue')
def booked_revenue_endpoint():
    return jsonify(get_comparison_data(get_booked_revenue_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/new-storage-customers')
def new_storage_customers_endpoint():
    return jsonify(get_comparison_data(get_new_storage_customers_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/repeat-storage-customers')
def repeat_storage_customers_endpoint():
    return jsonify(get_comparison_data(get_repeat_storage_customers_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/100k-deals')
def deals_100k_endpoint():
    return jsonify(get_comparison_data(get_large_deals_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region'), amount_threshold=100000))

@app.route('/api/data/200k-deals')
def deals_200k_endpoint():
    return jsonify(get_comparison_data(get_large_deals_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region'), amount_threshold=200000))

@app.route('/api/data/current-open-commit')
def current_open_commit_endpoint():
    return jsonify(get_comparison_data(get_current_open_commit_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/current-forecast')
def current_forecast_endpoint():
    return jsonify(get_comparison_data(get_current_forecast_data, request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/rep-performance')
def rep_performance_endpoint():
    return jsonify(get_rep_performance_data(request.args.get('start_date'), request.args.get('end_date')))

@app.route('/api/data/sdr-generated-opps-chart')
def sdr_generated_opps_chart_endpoint():
    return jsonify(get_sdr_generated_opps_chart_data(request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/sdr-emails')
def sdr_emails_endpoint():
    return jsonify(get_sdr_activity_data(request.args.get('start_date'), request.args.get('end_date'), 'Email'))

@app.route('/api/data/sdr-calls')
def sdr_calls_endpoint():
    return jsonify(get_sdr_activity_data(request.args.get('start_date'), request.args.get('end_date'), 'Call'))

@app.route('/api/data/sdr-meetings')
def sdr_meetings_endpoint():
    return jsonify(get_sdr_activity_data(request.args.get('start_date'), request.args.get('end_date'), 'Meeting'))

@app.route('/api/data/ml-predictions')
def ml_predictions_endpoint():
    months_lookback = request.args.get('months_lookback', '18')
    opp_type = request.args.get('opportunity_type', 'New Business')
    return jsonify(get_ml_predictions(months_lookback, opp_type))

@app.route('/api/filters/sales-regions')
def sales_regions_endpoint():
    return jsonify(get_sales_regions())

@app.route('/')
def dashboard_page():
    return render_template('index.html')

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
