import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import pandas as pd
from simple_salesforce import Salesforce
import plotly.graph_objects as go

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

# --- DATA FETCHING AND TRANSFORMATION FUNCTIONS ---

def get_forecast_data(start_date, end_date, sales_region):
    """ Fetches and transforms the forecast data based on filters. """
    if not sf_connection: return {}

    where_clauses = ["Primary_Product__c != 'Servers'"]
    if start_date:
        where_clauses.append(f"CloseDate >= {start_date}")
    if end_date:
        where_clauses.append(f"CloseDate <= {end_date}")
    
    # THE FIX IS HERE: Handle a comma-separated list of regions for an IN clause.
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        # Properly format for a SOQL IN clause: ('Region1','Region2',...)
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")
    
    soql_query = f"""
        SELECT Amount, Sales_Region__c, ForecastCategoryName
        FROM Opportunity
        WHERE {' AND '.join(where_clauses)}
    """
    
    try:
        query_result = sf_connection.query_all(soql_query)
        df = pd.DataFrame(query_result['records']).drop(columns=['attributes'])
    except Exception as e:
        print(f"❌ Forecast Query Error: {e}")
        return {}

    if df.empty: return {}
        
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    pivot_df = df.groupby(['Sales_Region__c', 'ForecastCategoryName'])['Amount'].sum().unstack(fill_value=0)
    region_summary = df.groupby('Sales_Region__c').agg(Opp_Count=('Amount', 'count'), Total_Amount=('Amount', 'sum'))
    sorted_regions = region_summary.sort_values(by='Total_Amount', ascending=False).index
    pivot_df = pivot_df.reindex(sorted_regions)
    region_summary = region_summary.reindex(sorted_regions)
    forecast_order = ['Pipeline', 'Best Case', 'Commit', 'Closed']
    existing_columns = [col for col in forecast_order if col in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]

    return {
        'regions': pivot_df.index.tolist(),
        'categories': pivot_df.columns.tolist(),
        'bar_data': {col: pivot_df[col].tolist() for col in pivot_df.columns},
        'line_data': {'regions': region_summary.index.tolist(), 'counts': region_summary['Opp_Count'].tolist()}
    }

def get_bookings_data(start_date, end_date, sales_region):
    """ Fetches and transforms bookings data by month based on filters. """
    if not sf_connection: return {}
    
    booking_products = "('TrueNAS', 'TrueNAS Mini', 'TrueCommand', 'TrueRack', 'TrueFlex')"
    
    where_clauses = [
        "StageName = 'Closed Won'",
        f"Primary_Product__c IN {booking_products}"
    ]
    if start_date:
        where_clauses.append(f"CloseDate >= {start_date}")
    if end_date:
        where_clauses.append(f"CloseDate <= {end_date}")
        
    # THE FIX IS HERE: Handle a comma-separated list of regions for an IN clause.
    if sales_region and sales_region != 'All':
        regions_list = sales_region.split(',')
        formatted_regions = "','".join(regions_list)
        where_clauses.append(f"Sales_Region__c IN ('{formatted_regions}')")

    soql_query = f"""
        SELECT CALENDAR_YEAR(CloseDate) c_year, CALENDAR_MONTH(CloseDate) c_month, SUM(Amount) totalAmount
        FROM Opportunity
        WHERE {' AND '.join(where_clauses)}
        GROUP BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate)
        ORDER BY CALENDAR_YEAR(CloseDate), CALENDAR_MONTH(CloseDate)
    """
    
    try:
        query_result = sf_connection.query_all(soql_query)
        df = pd.DataFrame(query_result['records'])
    except Exception as e:
        print(f"❌ Bookings Query Error: {e}")
        return {}
        
    if df.empty: return {}

    df['label'] = pd.to_datetime(
        df['c_year'].astype(str) + '-' + df['c_month'].astype(str) + '-01'
    ).dt.strftime('%b %Y')

    return {
        'labels': df['label'].tolist(),
        'amounts': df['totalAmount'].tolist()
    }

def get_sales_regions():
    """ Fetches a unique list of sales regions to populate the filter dropdown. """
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
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    sales_region = request.args.get('sales_region')
    data = get_forecast_data(start_date, end_date, sales_region)
    return jsonify(data)

@app.route('/api/data/bookings')
def bookings_data_endpoint():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    sales_region = request.args.get('sales_region')
    data = get_bookings_data(start_date, end_date, sales_region)
    return jsonify(data)

@app.route('/api/filters/sales-regions')
def sales_regions_endpoint():
    regions = get_sales_regions()
    return jsonify(regions)

@app.route('/')
def dashboard_page():
    return render_template('index.html')

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

