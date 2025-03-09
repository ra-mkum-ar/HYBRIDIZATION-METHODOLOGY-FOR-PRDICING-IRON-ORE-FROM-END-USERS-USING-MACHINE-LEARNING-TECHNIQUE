import serial
import time
import requests
import json

# Thinger.io Credentials
THINGER_USERNAME = "your_thinger_username"
DEVICE_ID = "Ore_Tracker"
DEVICE_TOKEN = "your_device_token"

# API Endpoint for sending data
THINGER_API_URL = f"https://api.thinger.io/v3/users/{THINGER_USERNAME}/devices/{DEVICE_ID}"

# Serial Port Configuration (Change based on your device)
GPS_PORT = "/dev/ttyUSB1"  # GPS Module Serial Port
RFID_PORT = "/dev/ttyUSB0"  # RFID Reader Serial Port
BAUD_RATE = 9600

# Connect to GPS & RFID Readers
try:
    gps_ser = serial.Serial(GPS_PORT, BAUD_RATE, timeout=1)
    rfid_ser = serial.Serial(RFID_PORT, BAUD_RATE, timeout=1)
    print("GPS & RFID Readers Connected!")
except Exception as e:
    print("Error connecting to GPS/RFID:", e)
    gps_ser = None
    rfid_ser = None

# Function to read GPS data
def read_gps():
    if gps_ser:
        try:
            line = gps_ser.readline().decode("utf-8").strip()
            if "$GPGGA" in line:
                parts = line.split(",")
                if len(parts) > 5:
                    lat = float(parts[2]) / 100
                    lon = float(parts[4]) / 100
                    return lat, lon
        except Exception as e:
            print("GPS Error:", e)
    return None, None

# Function to read RFID data
def read_rfid():
    if rfid_ser:
        try:
            tag = rfid_ser.readline().decode("utf-8").strip()
            return tag
        except Exception as e:
            print("RFID Error:", e)
    return None

# Function to send data to Thinger.io
def send_to_thinger(lat, lon, rfid_tag):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEVICE_TOKEN}"}
    data = {"latitude": lat, "longitude": lon, "rfid": rfid_tag}

    try:
        response = requests.post(f"{THINGER_API_URL}/data", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print(f"Data Sent: GPS ({lat}, {lon}) | RFID: {rfid_tag}")
        else:
            print(f"Failed to send data: {response.status_code}, {response.text}")
    except Exception as e:
        print("Error sending data:", e)

# Continuous tracking
print("Tracking ore movement... Press Ctrl+C to stop.")
try:
    while True:
        tag = read_rfid()
        lat, lon = read_gps()

        if tag and lat and lon:
            send_to_thinger(lat, lon, tag)

        time.sleep(2)  # Prevent excessive data sending
except KeyboardInterrupt:
    print("Tracking Stopped.")
