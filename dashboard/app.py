import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AetherSea",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #07111f;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #0b1727;
    border-right: 1px solid #1f2f46;
}

h1, h2, h3, h4 {
    color: white;
}

.metric-card {
    background: rgba(20, 30, 48, 0.85);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.status-online {
    color: #00ff9f;
    font-weight: bold;
}

.small-text {
    color: #9aa7b8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("AetherSea Control")

st.sidebar.markdown("---")

region = st.sidebar.selectbox(
    "Ocean Region",
    [
        "Arabian Sea",
        "Bay of Bengal",
        "Indian Ocean"
    ]
)

threshold = st.sidebar.slider(
    "Current Threshold",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1
)

st.sidebar.markdown("---")

st.sidebar.markdown("### System Status")

st.sidebar.success("HYCOM Connected")
st.sidebar.success("Ocean Intelligence Active")
st.sidebar.info("DBSCAN Offline")
st.sidebar.info("Routing Engine Offline")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🌊 AetherSea Maritime Intelligence System")

st.markdown(
    """
    <div class='small-text'>
    Real-Time Ocean Monitoring • Marine Debris Intelligence • AI Maritime Operations
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# TOP METRICS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Active Region</h3>
        <h2>Arabian Sea</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Ocean Status</h3>
        <h2 class="status-online">LIVE</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Potential Hotspots</h3>
        <h2>12</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Current Strength</h3>
        <h2>1.42 m/s</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------
# SAMPLE MAP DATA
# ---------------------------------------------------

data = pd.DataFrame({
    "lat": [15, 18, 22],
    "lon": [65, 70, 75],
    "label": [
        "Potential Debris Zone A",
        "Potential Debris Zone B",
        "Potential Debris Zone C"
    ]
})

# ---------------------------------------------------
# PLOTLY MAP
# ---------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scattermapbox(
        lat=data["lat"],
        lon=data["lon"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=16
        ),
        text=data["label"],
        hoverinfo="text"
    )
)

# ---------------------------------------------------
# MAP LAYOUT
# ---------------------------------------------------

fig.update_layout(

    mapbox_style="carto-darkmatter",

    mapbox=dict(
        center=dict(
            lat=18,
            lon=70
        ),
        zoom=3
    ),

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    height=700,

    paper_bgcolor="#07111f",
    plot_bgcolor="#07111f"
)

# ---------------------------------------------------
# MAIN GRID LAYOUT
# ---------------------------------------------------

left, right = st.columns([3, 1])

with left:

    st.subheader("Live Maritime Intelligence Map")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("Mission Intelligence")

    st.markdown("""
    <div class="metric-card">
        <h4>HYCOM Feed</h4>
        <p class="status-online">CONNECTED</p>

        <h4>Satellite Layer</h4>
        <p>PENDING</p>

        <h4>Cluster Detection</h4>
        <p>PENDING</p>

        <h4>AI Dispatch Agent</h4>
        <p>PENDING</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="metric-card">
        <h4>Operational Notes</h4>

        <p>
        Ocean current analysis indicates several
        moderate-flow convergence structures
        within the Arabian Sea operational region.
        </p>

        <p>
        Further satellite validation required
        before debris cluster confirmation.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "AetherSea • AI-Powered Maritime Environmental Intelligence Platform"
)