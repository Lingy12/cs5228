import math
import pandas

radius = 6371


def degree2radians(degree):
    return degree * math.pi / 180


def air_distance(origin_lat: str, origin_lon: str, destination_lat: str, destination_lon: str):
    oLat = float(origin_lat)
    oLon = float(origin_lon)
    dLat = float(destination_lat)
    dLon = float(destination_lon)
    degressLat = degree2radians(dLat - oLat)
    degressLon = degree2radians(dLon - oLon)
    a = math.sin(degressLat / 2) * math.sin(degressLat / 2) + math.cos(degree2radians(dLat)) * \
        math.sin(degressLon / 2) * math.sin(degressLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c * 1000
    return d


# threshold unit is in meters
def count_close_locations(df, threshold, lat, lon):
    result = df.apply(lambda x: air_distance(lat, lon, x["latitude"], x["longitude"]), axis=1)
    count = (result <= threshold).sum()
    return count

def average_price_in_range(df, threshold, lat, lon):
    # df = pandas.read_csv("../data/train.csv")
    df["primary_school_distance"] = df.apply(lambda x: air_distance(lat, lon, x["latitude"], x["longitude"]), axis=1)
    df = df[df["primary_school_distance"] <= threshold]
    avg = (df["monthly_rent"] / df["floor_area_sqm"]).mean()
    return avg

d = pandas.read_csv("../data/train.csv")
dist = 2000
primary_school = pandas.read_csv("../data/auxiliary-data/sg-primary-schools.csv")
primary_school["avg_price"] = primary_school.apply(lambda x: average_price_in_range(d, dist, x["latitude"], x["longitude"]), axis=1)
primary_school.to_csv(f"../data/primary_school_with_{dist/1000}_avg_rent.csv")

existing_mrt = pandas.read_csv("../data/auxiliary-data/sg-mrt-existing-stations.csv")
existing_mrt["avg_price"] = existing_mrt.apply(lambda x: average_price_in_range(d, dist, x["latitude"], x["longitude"]), axis=1)
existing_mrt.to_csv(f"../data/existing_mrt_with_{dist/1000}_avg_rent.csv")

planning_mrt = pandas.read_csv("../data/auxiliary-data/sg-mrt-planned-stations.csv")
planning_mrt["avg_price"] = planning_mrt.apply(lambda x: average_price_in_range(d, dist, x["latitude"], x["longitude"]), axis=1)
planning_mrt.to_csv(f"../data/planning_mrt_with_{dist/1000}_avg_rent.csv")

shopping_malls = pandas.read_csv("../data/auxiliary-data/sg-shopping-malls.csv")
shopping_malls["avg_price"] = shopping_malls.apply(lambda x: average_price_in_range(d, dist, x["latitude"], x["longitude"]), axis=1)
shopping_malls.to_csv(f"../data/shopping_malls_with_{dist/1000}_avg_rent.csv")


