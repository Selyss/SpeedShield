import pandas as pd
import json

csv = pd.read_csv("existing_cameras.csv")

def extract_coords(geometry):
    try:
        geo = json.loads(geometry)
        coords = geo["coordinates"][0]
        return pd.Series({"longitude": coords[0], "latitude": coords[1]})
    except Exception as e:
        print("Failed to parse geometry: ", geometry, "Error: ", e)
        return pd.Series({"longitude": None, "latitude": None})

coord = csv["geometry"].apply(extract_coords)

csv = pd.concat([csv, coord], axis=1)

csv.to_csv("existing_cameras_processed.csv", index=False)