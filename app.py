import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import pandas as pd
from simple_salesforce import Salesforce
import plotly.graph_objects as go
from datetime import datetime
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
    """
    Recursively converts numpy number types in a dictionary or list
    to standard Python types to ensure they are JSON serializable.
    """
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
    soql_query = f"SELECT CALENDAR_YEAR(CloseDate) c_year, CALENDAR_MONTH(CloseDate) c_month, SUM(Amount) totalAmount FROM Opportunity WHERE {' AND '.join(where_clauses)} GROUP BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate) ORDER BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate)"
    try:
        query_result = sf_connection.query_all(soql_query)
        df = pd.DataFrame(query_result['records'])
    except Exception as e:
        print(f"❌ Bookings Query Error: {e}")
        return {}
    if df.empty: return {}
    df['label'] = pd.to_datetime(df['c_year'].astype(str) + '-' + df['c_month'].astype(str) + '-01').dt.strftime('%b %Y')
    return {'labels': df['label'].tolist(), 'amounts': df['totalAmount'].tolist()}

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

def get_sdr_activity_data(start_date, end_date, activity_type):
    if not sf_connection: return {}
    sdr_names = "('Sheen Trisal', 'Brandon Mazikowski', 'Hayden Barcelos', 'Jaylen Macias-Matsuura')"
    where_clauses = [f"Owner.Name IN {sdr_names}", f"Type = '{activity_type}'", f"ActivityDate <= {datetime.now().strftime('%Y-%m-%d')}"]
    if activity_type == 'Email': where_clauses.append("Subject LIKE '%Gong%'")
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
    pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y-%m').strftime('%b %Y')
    return {'labels': pivot_df.index.tolist(), 'sdr_data': {col: pivot_df[col].tolist() for col in pivot_df.columns}, 'sdr_names': pivot_df.columns.tolist()}

def get_ml_predictions():
    if not sf_connection: return {"error": "Salesforce connection not available."}
    
    TARGET_OBJECT_ML = 'Opportunity'
    OPPORTUNITY_TYPE_FILTER = 'New Business'
    PRIMARY_PRODUCT_FILTER = 'TrueNAS'
    MONTHS_LOOKBACK = 18
    TARGET_VARIABLE = 'IsWon'
    
    features_to_select = ['Amount', 'LeadSource', 'Type', 'Sales_Region__c', 'CreatedDate', 'CloseDate', 'IsWon', 'IsClosed', 'StageName']
    
    eighteen_months_ago_str = (datetime.now() - relativedelta(months=MONTHS_LOOKBACK)).strftime('%Y-%m-%d')
    soql_ml_query = f"SELECT {', '.join(features_to_select)} FROM {TARGET_OBJECT_ML} WHERE Type = '{OPPORTUNITY_TYPE_FILTER}' AND Primary_Product__c = '{PRIMARY_PRODUCT_FILTER}' AND IsClosed = TRUE AND CloseDate >= {eighteen_months_ago_str}"
    
    try:
        print("⏳ Fetching data for ML Model...")
        ml_records = sf_connection.query_all_iter(soql_ml_query)
        df_ml = pd.DataFrame(list(ml_records))
        if 'attributes' in df_ml.columns:
            df_ml = df_ml.drop(columns=['attributes'])
        if df_ml.empty:
            return {"error": "No opportunities found for ML model training."}
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
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("⏳ Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("✅ Model trained.")
    
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # THE FIX IS HERE: Changed from .head(10) to .head(30) to show more features.
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
    return jsonify(get_goal_data(request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/booked-revenue')
def booked_revenue_endpoint():
    return jsonify(get_booked_revenue_data(request.args.get('start_date'), request.args.get('end_date'), request.args.get('sales_region')))

@app.route('/api/data/sdr-emails')
def sdr_emails_endpoint():
    return jsonify(get_sdr_activity_data(request.args.get('start_date'), request.args.get('end_date'), 'Email'))

@app.route('/api/data/sdr-calls')
def sdr_calls_endpoint():
    return jsonify(get_sdr_activity_data(request.args.get('start_date'), request.args.get('end_date'), 'Call'))

@app.route('/api/data/ml-predictions')
def ml_predictions_endpoint():
    return jsonify(get_ml_predictions())

@app.route('/api/filters/sales-regions')
def sales_regions_endpoint():
    return jsonify(get_sales_regions())

@app.route('/')
def dashboard_page():
    return render_template('index.html')

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

