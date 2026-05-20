import math

# ---------------------------------------------------
# EARTH RADIUS
# ---------------------------------------------------

EARTH_RADIUS_KM = 6371

# ---------------------------------------------------
# HAVERSINE DISTANCE FUNCTION
# ---------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):

    # Convert degrees to radians

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Coordinate differences

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine Formula

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    distance = EARTH_RADIUS_KM * c

    return distance

# ---------------------------------------------------
# HOME PORT
# ---------------------------------------------------

home_port = {
    "name": "Mumbai Port",
    "lat": 18.96,
    "lon": 72.82
}

# ---------------------------------------------------
# DETECTED HOTSPOTS
# ---------------------------------------------------

hotspots = [

    {
        "cluster_id": 0,
        "center_lat": 22.405,
        "center_lon": 72.50
    },

    {
        "cluster_id": 1,
        "center_lat": 19.20,
        "center_lon": 70.10
    },

    {
        "cluster_id": 2,
        "center_lat": 24.80,
        "center_lon": 74.90
    }

]

# ---------------------------------------------------
# GREEDY ROUTE OPTIMIZATION
# ---------------------------------------------------

current_lat = home_port["lat"]
current_lon = home_port["lon"]

remaining_hotspots = hotspots.copy()

optimized_route = []

total_distance = 0

print("\nAetherSea Route Optimization Started\n")

while remaining_hotspots:

    nearest_hotspot = None

    nearest_distance = float("inf")

    # ---------------------------------------------------
    # FIND NEAREST HOTSPOT
    # ---------------------------------------------------

    for hotspot in remaining_hotspots:

        distance = haversine_distance(

            current_lat,
            current_lon,

            hotspot["center_lat"],
            hotspot["center_lon"]

        )

        if distance < nearest_distance:

            nearest_distance = distance

            nearest_hotspot = hotspot

    # ---------------------------------------------------
    # ADD TO ROUTE
    # ---------------------------------------------------

    optimized_route.append({

        "cluster_id": nearest_hotspot["cluster_id"],

        "lat": nearest_hotspot["center_lat"],

        "lon": nearest_hotspot["center_lon"],

        "distance_from_previous_km": round(
            nearest_distance,
            2
        )

    })

    total_distance += nearest_distance

    # ---------------------------------------------------
    # MOVE SHIP POSITION
    # ---------------------------------------------------

    current_lat = nearest_hotspot["center_lat"]

    current_lon = nearest_hotspot["center_lon"]

    # ---------------------------------------------------
    # REMOVE VISITED HOTSPOT
    # ---------------------------------------------------

    remaining_hotspots.remove(nearest_hotspot)

# ---------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------

print("Optimized Cleanup Route:\n")

for step_number, point in enumerate(optimized_route, start=1):

    print(f"Step {step_number}")

    print(f"Cluster ID: {point['cluster_id']}")

    print(f"Coordinates: ({point['lat']}, {point['lon']})")

    print(
        f"Travel Distance: "
        f"{point['distance_from_previous_km']} km\n"
    )

print("----------------------------------")

print(
    f"\nTotal Maritime Route Distance: "
    f"{round(total_distance, 2)} km"
)