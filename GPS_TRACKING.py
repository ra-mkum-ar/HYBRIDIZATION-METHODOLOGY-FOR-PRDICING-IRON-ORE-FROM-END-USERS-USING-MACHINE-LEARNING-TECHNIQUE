import serial  # For GPS & RFID communication
import pandas as pd
import time
from geopy.geocoders import Nominatim
import folium

# Configure Serial Ports
GPS_PORT = "/dev/ttyUSB1"  # Adjust based on GPS module connection
RFID_PORT = "/dev/ttyUSB0"  # Adjust for RFID reader
BAUD_RATE = 9600

# Connect to GPS & RFID Reader
try:
    gps_ser = serial.Serial(GPS_PORT, BAUD_RATE, timeout=1)
    rfid_ser = serial.Serial(RFID_PORT, BAUD_RATE, timeout=1)
    print("GPS & RFID Readers Connected Successfully!")
except Exception as e:
    print("Error connecting to GPS/RFID:", e)
    gps_ser = None
    rfid_ser = None

# File to log GPS & RFID data
GPS_LOG_FILE = "gps_rfid_tracking.csv"

# Function to read GPS data
def read_gps():
    if gps_ser:
        try:
            line = gps_ser.readline().decode("utf-8").strip()
            if "$GPGGA" in line:  # GPGGA sentence contains GPS coordinates
                parts = line.split(",")
                if len(parts) > 5:
                    lat = float(parts[2]) / 100
                    lon = float(parts[4]) / 100
                    return lat, lon
        except Exception as e:
            print("Error reading GPS:", e)
    return None, None

# Function to read RFID tag
def read_rfid():
    if rfid_ser:
        try:
            rfid_data = rfid_ser.readline().decode("utf-8").strip()
            return rfid_data
        except Exception as e:
            print("Error reading RFID:", e)
    return None

# Function to log GPS & RFID data
def log_data(tag_id, lat, lon):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert coordinates to location
    geolocator = Nominatim(user_agent="ore_tracking_system")
    location = geolocator.reverse(f"{lat}, {lon}") if lat and lon else "Unknown"

    # Read or create log file
    try:
        df = pd.read_csv(GPS_LOG_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Timestamp", "RFID_Tag", "Latitude", "Longitude", "Location"])

    # Append new data
    new_entry = pd.DataFrame([[timestamp, tag_id, lat, lon, location]], 
                             columns=["Timestamp", "RFID_Tag", "Latitude", "Longitude", "Location"])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save back to CSV
    df.to_csv(GPS_LOG_FILE, index=False)
    print(f"Logged: {timestamp} | RFID: {tag_id} | Location: {location}")

# Function to visualize GPS movement on a map
def generate_map():
    try:
        df = pd.read_csv(GPS_LOG_FILE)
        if df.empty:
            print("No GPS data available to generate a map.")
            return

        # Create a map centered at the last recorded location
        last_lat, last_lon = df.iloc[-1][["Latitude", "Longitude"]]
        gps_map = folium.Map(location=[last_lat, last_lon], zoom_start=10)

        # Add markers for each location
        for _, row in df.iterrows():
            folium.Marker([row["Latitude"], row["Longitude"]], 
                          popup=f'RFID: {row["RFID_Tag"]}<br>Time: {row["Timestamp"]}'
                         ).add_to(gps_map)

        # Save map as HTML
        gps_map.save("gps_tracking_map.html")
        print("GPS Tracking Map Saved: gps_tracking_map.html")
    except Exception as e:
        print("Error generating map:", e)

# Continuous RFID & GPS scanning
print("Tracking Ore Movement... Press Ctrl+C to stop.")
try:
    while True:
        tag = read_rfid()
        lat, lon = read_gps()

        if tag and lat and lon:
            print(f"RFID: {tag} | GPS: ({lat}, {lon})")
            log_data(tag, lat, lon)

        time.sleep(2)  # Prevent excessive reading
except KeyboardInterrupt:
    print("Tracking Stopped.")
    generate_map()  # Generate GPS tracking map when stopped
