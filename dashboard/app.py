import streamlit as st
import pandas as pd
import plotly.express as px
import time
import datetime
import requests

API_BASE_URL = "http://localhost:8000"
STORE_ID = "STORE_BLR_002"

st.set_page_config(page_title="Store Intelligence", layout="wide", page_icon="🛍️", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] {
        background-color: #1e1e1e; padding: 15px; border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-left: 5px solid #4CAF50;
    }
    .last-updated { color: #888888; font-size: 0.9em; text-align: right; margin-top: 20px; }
    .anomaly-card { background-color: #3d1b1b; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

col_title, col_time = st.columns([3, 1])
with col_title:
    st.title("🛍️ Store Intelligence Dashboard")
    st.markdown("Real-time AI CCTV Analytics Platform")
with col_time:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"<p class='last-updated'>Last Updated: {now_str}</p>", unsafe_allow_html=True)

try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=2)
    api_healthy = health_res.status_code == 200
except Exception:
    api_healthy = False

if not api_healthy:
    st.error("Cannot connect to Intelligence API. Ensure backend is running.")
    st.stop()

def fetch_data():
    try:
        metrics = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/metrics").json()
        funnel = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/funnel").json()
        heatmap = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/heatmap").json()
        anomalies = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/anomalies").json()
        events = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/events?limit=20").json()
        return metrics, funnel, heatmap, anomalies, events
    except Exception as e:
        return None, None, None, None, None

metrics, funnel, heatmap, anomalies, events = fetch_data()


st.json(metrics)

if not metrics:
    st.info("No data available yet or API error.")
else:
    st.markdown("---")
    st.markdown("---")
    st.subheader("📊 Live Key Performance Indicators")
    
    # Row 1 of KPIs
   # Executive KPI Row
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        "👥 Visitors",
        metrics.get("unique_visitors", 0)
    )

with k2:
    st.metric(
        "💰 Revenue",
        f"₹{metrics.get('revenue', 0):,.0f}"
    )

with k3:
    st.metric(
        "📈 Conversion Rate",
        f"{metrics.get('conversion_rate', 0):.1f}%"
    )

with k4:
    st.metric(
        "🛒 Avg Basket",
        f"₹{metrics.get('avg_basket', 0):,.0f}"
    )

with k5:
    st.metric(
        "🚪 Queue Abandon %",
        f"{metrics.get('queue_abandon_percent', 0):.1f}%"
    )

with k6:
    st.metric(
        "🏃 Queue Depth",
        metrics.get("queue_depth", 0)
    )

st.markdown("---")
    
anomaly_list = anomalies.get("anomalies", [])
if anomaly_list:
        st.subheader("🚨 Active Alerts")
        for a in anomaly_list:
            st.markdown(f"<div class='anomaly-card'><strong>{a['severity']} - {a['type']}</strong><br>{a['message']}<br><em>Action: {a['suggested_action']}</em></div>", unsafe_allow_html=True)
            
st.markdown("---")
    
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
        st.subheader("🛒 Conversion Funnel")
        f_data = funnel.get("funnel", {})
        if f_data.get("entries", 0) > 0:
            funnel_df = pd.DataFrame([
                {"Stage": "Entries", "Count": f_data.get("entries", 0)},
                {"Stage": "Zone Visits", "Count": f_data.get("zone_visits", 0)},
                {"Stage": "Billing Queue", "Count": f_data.get("billing_queue", 0)},
                {"Stage": "Purchases", "Count": f_data.get("purchases", 0)}
            ])
            fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', template='plotly_dark', title="Shopper Journey")
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("Not enough data for funnel.")
        
        # Display POS correlation details
        st.markdown("### 💳 POS Conversion Summary")
        st.markdown(f"**Converted Visitors:** {metrics.get('converted_visitors', 0)}")
        st.markdown(f"**Total Transactions:** {metrics.get('transactions', 0)}")
        st.markdown(f"**Avg Dwell per Zone:** {metrics.get('avg_dwell_per_zone_sec', 0)} seconds")
            
with chart_col2:
        st.subheader("🔥 Zone Heatmap")
        z_data = heatmap.get("zones", {})
        if z_data:
            zone_df = pd.DataFrame([
                {"Zone": k, "Normalized Visits": v["normalized_visits"], "Avg Dwell": v["avg_dwell_ms"]/1000}
                for k, v in z_data.items()
            ])
            fig_bar = px.bar(zone_df, x='Zone', y='Normalized Visits', color='Avg Dwell', 
                             title="Zone Activity (Color = Dwell Time)", template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No zone data available.")
            
        # Display Top Zones list
        st.markdown("### 🏆 Top Zones by Visits")
        top_zones = metrics.get("top_zones", [])
        if top_zones:
            top_zones_df = pd.DataFrame(top_zones)
            top_zones_df.columns = ["Zone ID", "Visits"]
            st.dataframe(top_zones_df, use_container_width=True, hide_index=True)
        else:
            st.info("No top zone visits data.")

st.subheader("🕒 Recent Events Live Feed")
if isinstance(events, dict):
        if "detail" in events:
            st.warning(f"Error fetching events: {events['detail']}")
        else:
            df_events = pd.DataFrame([events])
            st.dataframe(df_events, use_container_width=True, hide_index=True)
elif isinstance(events, list) and len(events) > 0:
        df_events = pd.DataFrame(events)
        st.dataframe(df_events, use_container_width=True, hide_index=True)
else:
        st.info("No recent events.")

time.sleep(5)
st.rerun()
