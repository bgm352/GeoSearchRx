import streamlit as st
import pandas as pd
import os
from pytrends.request import TrendReq
import openai
import requests
import tempfile
import json

# For Gemini
import google.generativeai as genai

# For Google Search Console
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- Sidebar: API Key Input ---
st.set_page_config(page_title="Health System Search Insights", layout="wide")
st.title("Health System Search Insights Dashboard")

st.sidebar.header("🔑 API Key Input")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_api_key")
gsc_json = st.sidebar.text_area("Google Search Console Service Account JSON (paste entire JSON)", key="gsc_api_key")
semrush_key = st.sidebar.text_input("SEMrush API Key", type="password", key="semrush_api_key")

if openai_key:
    st.session_state["OPENAI_API_KEY"] = openai_key
if gemini_key:
    st.session_state["GEMINI_API_KEY"] = gemini_key
if gsc_json:
    st.session_state["GSC_API_KEY"] = gsc_json
if semrush_key:
    st.session_state["SEMRUSH_API_KEY"] = semrush_key

st.sidebar.markdown("---")
st.sidebar.info("Enter your API keys above. They are only stored for your session and never shared.")

# --- Check for missing keys ---
missing_keys = [k for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GSC_API_KEY", "SEMRUSH_API_KEY"] if k not in st.session_state]
if missing_keys:
    st.warning(f"Please provide all required API keys: {', '.join(missing_keys)}")

# --- Data Input ---
st.sidebar.header("Data Sources & Settings")
keywords = st.sidebar.text_input("Enter keywords (comma-separated)", "cardiology, urgent care")
geo = st.sidebar.text_input("Geography (e.g., US, GB, etc.)", "US")
domain = st.sidebar.text_input("Domain for SEMrush", "mayoclinic.org")
site_url = st.sidebar.text_input("Google Search Console Site URL", "https://www.examplehealth.org")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# --- Google Trends via Pytrends ---
@st.cache_data
def get_google_trends(keywords, geo='US', timeframe='today 12-m'):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(keywords, geo=geo, timeframe=timeframe)
    df = pytrends.interest_over_time()
    df = df.reset_index()
    df = df.drop(columns=['isPartial'], errors='ignore')
    df = df.melt(id_vars=['date'], var_name='keyword', value_name='search_volume')
    return df

# --- Google Search Console API ---
def get_gsc_data(service_account_json, site_url, start_date, end_date):
    # Write the JSON to a temp file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
        tmp.write(service_account_json)
        tmp_path = tmp.name
    credentials = service_account.Credentials.from_service_account_file(
        tmp_path,
        scopes=['https://www.googleapis.com/auth/webmasters.readonly'],
    )
    service = build('searchconsole', 'v1', credentials=credentials)
    request = {
        'startDate': str(start_date),
        'endDate': str(end_date),
        'dimensions': ['query', 'country'],
        'rowLimit': 100
    }
    response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
    rows = response.get('rows', [])
    data = []
    for row in rows:
        data.append({
            'query': row['keys'][0],
            'country': row['keys'][1],
            'clicks': row['clicks'],
            'impressions': row['impressions'],
            'ctr': row['ctr'],
            'position': row['position']
        })
    os.remove(tmp_path)
    return pd.DataFrame(data)

# --- SEMrush API ---
def get_semrush_data(domain, api_key, database="us"):
    url = f"https://api.semrush.com/?type=domain_organic&key={api_key}&display_limit=10&export_columns=Ph,Po,Nq&domain={domain}&database={database}"
    response = requests.get(url)
    if response.status_code == 200:
        data = [line.split(';') for line in response.text.strip().split('\n')]
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    else:
        st.error("SEMrush API error")
        return pd.DataFrame()

# --- OpenAI Analysis ---
def run_openai_analysis(prompt, data, api_key):
    openai.api_key = api_key
    full_prompt = f"{prompt}\n\nData:\n{data.head(10).to_csv(index=False)}"
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message['content']

# --- Gemini Analysis ---
def run_gemini_analysis(prompt, data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"{prompt}\n\nData:\n{data.head(10).to_csv(index=False)}")
    return response.text

# --- Main Logic ---
if st.sidebar.button("Fetch Data & Analyze"):
    if missing_keys:
        st.error("Please enter all required API keys before proceeding.")
        st.stop()
    kw_list = [k.strip() for k in keywords.split(",")]

    st.subheader("Google Trends Data")
    try:
        trends_df = get_google_trends(kw_list, geo=geo)
        st.dataframe(trends_df)
    except Exception as e:
        st.error(f"Google Trends error: {e}")
        trends_df = pd.DataFrame()

    st.subheader("Google Search Console Data")
    try:
        gsc_df = get_gsc_data(st.session_state["GSC_API_KEY"], site_url, start_date, end_date)
        st.dataframe(gsc_df)
    except Exception as e:
        st.error(f"GSC error: {e}")
        gsc_df = pd.DataFrame()

    st.subheader("SEMrush Data")
    try:
        semrush_df = get_semrush_data(domain, st.session_state["SEMRUSH_API_KEY"])
        st.dataframe(semrush_df)
    except Exception as e:
        st.error(f"SEMrush error: {e}")
        semrush_df = pd.DataFrame()

    # Data Formatting Example
    st.subheader("Formatted SEO Data (Top Keywords)")
    if not trends_df.empty and not gsc_df.empty and not semrush_df.empty:
        merged = pd.merge(trends_df, gsc_df, left_on='keyword', right_on='query', how='outer')
        merged = pd.merge(merged, semrush_df, left_on='keyword', right_on='Ph', how='outer')
        st.dataframe(merged.head(20))

        # AI Insights
        st.subheader("AI Insights & Recommendations")
        prompt = "Analyze the following SEO data for market opportunities, keyword gaps, and actionable recommendations for a health system marketing team."
        with st.expander("OpenAI GPT-4 Analysis"):
            st.write(run_openai_analysis(prompt, merged, st.session_state["OPENAI_API_KEY"]))
        with st.expander("Gemini Analysis"):
            st.write(run_gemini_analysis(prompt, merged, st.session_state["GEMINI_API_KEY"]))
    else:
        st.warning("Not all data sources returned data. Please check your inputs and API keys.")

st.info("Set up your API keys in the sidebar. This is a pilot app for live SEO data and AI-powered insights.")
