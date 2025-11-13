import streamlit as st
import requests
import folium
from math import radians, sin, cos, sqrt, atan2

# =========================================
# 🚗 Live Route Traffic + Nearby Incidents (TomTom + Bing API)
# =========================================

st.set_page_config(page_title="🗺️ Live Route Traffic", layout="centered")

# === API KEYS ===
TOMTOM_API_KEY = "vVzbNRaFcTDVcNFbUY3agB3O8Srt7LKw"

# Bing Maps API key — free tier (no billing)
BING_API_KEY = "ArbiLKl9Ut-dummy-demo-key-123456"  # You can get your own from https://www.bingmapsportal.com/

# === PAGE TITLE ===
st.title("🗺️ Live Route Traffic (TomTom + Bing Maps)")
st.write("View **real-time congestion**, **accidents**, and **road closures** along your route 🚦")

# === CSS Styling ===
st.markdown("""
<style>
h1, h2, h3 {text-align: center; color: #00b4d8;}
.stButton>button {
    background-color: #00b4d8;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
}
.stButton>button:hover { background-color: #0077b6; }
.clear-btn > button {
    background-color: #6c757d !important;
}
.clear-btn > button:hover {
    background-color: #495057 !important;
}
.refresh-btn > button {
    background-color: #ffb703 !important;
    color: black !important;
}
.refresh-btn > button:hover {
    background-color: #f48c06 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# === SESSION STATE INIT ===
for key in ["route", "map_html", "summary", "route_points", "src", "dest", "incidents"]:
    if key not in st.session_state:
        st.session_state[key] = None

# === Helper Functions ===
def geocode(location):
    """Convert location name into coordinates using TomTom API."""
    try:
        url = f"https://api.tomtom.com/search/2/geocode/{location}.json?key={TOMTOM_API_KEY}"
        data = requests.get(url, timeout=10).json()
        if data.get("results"):
            return data["results"][0]["position"]
    except Exception as e:
        st.error(f"⚠️ Geocoding error: {e}")
    return None


def get_route(src, dest):
    """Fetch route between two coordinates."""
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{src['lat']},{src['lon']}:{dest['lat']},{dest['lon']}/json?traffic=true&key={TOMTOM_API_KEY}"
    )
    try:
        res = requests.get(url, timeout=10).json()
        if "routes" not in res:
            return None
        return res["routes"][0]
    except Exception as e:
        st.error(f"⚠️ Error fetching route: {e}")
        return None


def get_traffic(points):
    """Fetch congestion levels along route."""
    data = []
    checkpoints = points[::max(1, len(points)//6)]
    for p in checkpoints:
        lat, lon = p["latitude"], p["longitude"]
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&unit=KMPH&key={TOMTOM_API_KEY}"
        try:
            resp = requests.get(url, timeout=10).json().get("flowSegmentData", {})
        except:
            continue
        if not resp:
            continue
        curr, free = resp.get("currentSpeed", 0), resp.get("freeFlowSpeed", 0)
        if free == 0:
            continue
        congestion = round((1 - curr / free) * 100, 1)
        if congestion < 25:
            color, label = "green", "🟢 Low"
        elif congestion < 60:
            color, label = "orange", "🟠 Moderate"
        else:
            color, label = "red", "🔴 High"
        data.append((lat, lon, color, label, curr, free))
    return data


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates."""
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_route_bbox(route_points, pad_deg=0.09):
    """Create a bounding box (~10 km buffer)."""
    lats = [p["latitude"] for p in route_points]
    lons = [p["longitude"] for p in route_points]
    return min(lons) - pad_deg, min(lats) - pad_deg, max(lons) + pad_deg, max(lats) + pad_deg


def get_local_incidents(route_points):
    """Fetch nearby incidents from Bing Maps API."""
    min_lon, min_lat, max_lon, max_lat = get_route_bbox(route_points)
    url = (
        f"http://dev.virtualearth.net/REST/v1/Traffic/Incidents/"
        f"{min_lat},{min_lon},{max_lat},{max_lon}"
        f"?key={BING_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        nearby = []
        for res in data.get("resourceSets", []):
            for inc in res.get("resources", []):
                title = inc.get("description", "Unknown incident")
                severity = inc.get("severity", 1)
                incident_type = inc.get("type", 1)
                point = inc.get("point", {}).get("coordinates", [0, 0])
                lat, lon = point[0], point[1]

                # Filter incidents within 10 km of route
                min_dist = min(haversine(lat, lon, p["latitude"], p["longitude"]) for p in route_points)
                if min_dist <= 10:
                    icons = {
                        1: "🟢 Minor",
                        2: "🟠 Moderate",
                        3: "🔴 Major",
                        4: "⚫ Critical"
                    }
                    types = {
                        1: "Accident",
                        2: "Congestion",
                        3: "Disabled Vehicle",
                        4: "Mass Transit",
                        5: "Miscellaneous",
                        6: "Other News",
                        7: "Planned Event",
                        8: "Road Hazard",
                        9: "Construction",
                        10: "Alert",
                        11: "Weather"
                    }
                    nearby.append({
                        "type": f"{types.get(incident_type, 'Incident')} ({icons.get(severity)})",
                        "lat": lat,
                        "lon": lon,
                        "desc": title
                    })
        return nearby
    except Exception as e:
        st.warning(f"⚠️ Could not fetch Bing incidents: {e}")
        return []


def build_map(src, dest, points, traffic_data, incidents_list):
    """Render folium map with incidents."""
    m = folium.Map(location=(src["lat"], src["lon"]), zoom_start=12)

    folium.PolyLine([(p["latitude"], p["longitude"]) for p in points],
                    color="blue", weight=6).add_to(m)

    for lat, lon, color, label, curr, free in traffic_data:
        folium.CircleMarker(
            location=(lat, lon),
            radius=7, color=color, fill=True,
            popup=f"{label} ({curr}/{free} km/h)"
        ).add_to(m)

    for inc in incidents_list:
        folium.Marker(
            [inc["lat"], inc["lon"]],
            popup=f"<b>{inc['type']}</b><br>{inc['desc']}",
            icon=folium.Icon(color="red" if "Accident" in inc["type"] else "orange")
        ).add_to(m)

    folium.Marker((src["lat"], src["lon"]), popup="Source", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((dest["lat"], dest["lon"]), popup="Destination", icon=folium.Icon(color="red")).add_to(m)

    return m._repr_html_()


# === UI ===
if st.session_state.route is None:
    with st.form("route_form"):
        src_input = st.text_input("Enter Source (e.g., Silk Board, Bengaluru)")
        dest_input = st.text_input("Enter Destination (e.g., Yelahanka, Bengaluru)")
        submit = st.form_submit_button("🚗 Get Live Route")

        if submit:
            if not src_input or not dest_input:
                st.warning("⚠️ Please enter both source and destination.")
            else:
                src = geocode(src_input)
                dest = geocode(dest_input)
                if not src or not dest:
                    st.error("❌ Location not found.")
                else:
                    with st.spinner("Fetching route, traffic & incidents..."):
                        route = get_route(src, dest)
                        if not route:
                            st.error("❌ Could not find route.")
                        else:
                            points = route["legs"][0]["points"]
                            dist = route["summary"]["lengthInMeters"] / 1000
                            time_min = route["summary"]["travelTimeInSeconds"] / 60
                            traffic = get_traffic(points)
                            local_incidents = get_local_incidents(points)
                            map_html = build_map(src, dest, points, traffic, local_incidents)

                            st.session_state.update({
                                "route": (src_input, dest_input),
                                "map_html": map_html,
                                "summary": [t[3] for t in traffic],
                                "route_points": points,
                                "src": src,
                                "dest": dest,
                                "incidents": local_incidents
                            })

                            st.success(f"✅ Route from {src_input} → {dest_input}")
                            st.metric("📏 Distance (km)", f"{dist:.2f}")
                            st.metric("⏱️ Duration (min)", f"{time_min:.1f}")


# === DISPLAY MAP ===
if st.session_state.map_html:
    src_input, dest_input = st.session_state.route
    st.markdown(f"## 🗺️ {src_input} → {dest_input}")
    st.components.v1.html(st.session_state.map_html, height=500)

    st.markdown("### 📊 Congestion Summary")
    st.write(" → ".join(st.session_state.summary))

    local_incidents = st.session_state.get("incidents", [])
    if local_incidents:
        st.markdown("### ⚠️ Nearby Incidents (via Bing Maps)")
        for i, inc in enumerate(local_incidents, 1):
            st.write(f"**{i}.** {inc['type']} – {inc['desc']}")
    else:
        st.success("✅ No active incidents near your route!")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("🔄 Refresh Live Data"):
            with st.spinner("Refreshing..."):
                traffic = get_traffic(st.session_state.route_points)
                local_incidents = get_local_incidents(st.session_state.route_points)
                st.session_state.map_html = build_map(
                    st.session_state.src, st.session_state.dest, st.session_state.route_points, traffic, local_incidents
                )
                st.session_state.summary = [t[3] for t in traffic]
                st.session_state.incidents = local_incidents
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear & Choose New Route"):
            for key in ["route", "map_html", "summary", "route_points", "src", "dest", "incidents"]:
                st.session_state[key] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr><center>Developed by <b>Nitesh & Team 🚀 | BMSIT</b></center>", unsafe_allow_html=True)
