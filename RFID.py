import serial  # For RFID reader communication
import pandas as pd
import time

# Serial port configuration (adjust based on your RFID reader)
RFID_PORT = "/dev/ttyUSB0"  # Change this for Windows: "COM3"
BAUD_RATE = 9600

# Connect to the RFID reader
try:
    ser = serial.Serial(RFID_PORT, BAUD_RATE, timeout=1)
    print("RFID Reader Connected Successfully!")
except Exception as e:
    print("Error connecting to RFID Reader:", e)
    ser = None

# CSV file to log RFID scans
RFID_LOG_FILE = "rfid_tracking_log.csv"

# Function to read RFID tag
def read_rfid():
    if ser:
        try:
            rfid_data = ser.readline().decode("utf-8").strip()
            return rfid_data
        except Exception as e:
            print("Error reading RFID:", e)
    return None

# Function to log RFID data
def log_rfid_data(tag_id):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the file exists, otherwise create it with headers
    try:
        df = pd.read_csv(RFID_LOG_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Timestamp", "RFID_Tag"])

    # Append new data
    new_entry = pd.DataFrame([[timestamp, tag_id]], columns=["Timestamp", "RFID_Tag"])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save back to CSV
    df.to_csv(RFID_LOG_FILE, index=False)
    print(f"RFID Tag {tag_id} logged at {timestamp}")

# Continuous RFID scanning
print("Waiting for RFID scans... Press Ctrl+C to stop.")
try:
    while True:
        tag = read_rfid()
        if tag:
            print(f"Detected RFID Tag: {tag}")
            log_rfid_data(tag)
        time.sleep(1)  # Delay to prevent excessive reading
except KeyboardInterrupt:
    print("RFID Tracking Stopped.")
