import folium
import pandas as pd
from datetime import datetime
from pyproj import Transformer

def visualize_bus_stations(data_filename: str = "data/data.csv"):
    map = folium.Map()
    df = pd.read_csv(data_filename)

    df = df.rename(columns={
        "Nom": "Name",
        "UTMx": "UTMx",
        "UTMy": "UTMy"
    })

    df = df.dropna(subset=["UTMx", "UTMy"])

    transformer = Transformer.from_crs("epsg:25831", "epsg:4326")
    df["lat"], df["lon"] = transformer.transform(df["UTMx"].values, df["UTMy"].values)

    m = folium.Map(location=[41.38, 2.17], zoom_start=13)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"{row["Name"]} ({row["Codi"]})",
            icon=folium.Icon(color="blue", icon="bus", prefix="fa")
        ).add_to(m)

    map_filename = f"maps/{data_filename.split("/")[-1].split(".")[0]}_{datetime.now().strftime("%Y%m%d%H%M")}_map.html"
    m.save(map_filename)
    print(f"Map saved to {map_filename}.")
