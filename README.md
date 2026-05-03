## How to open the dataset
1. Install HDFView
   Download HDFView from the official website (https://www.hdfgroup.org/download-hdfview/). Select the version matching your operating system (Windows, macOS, or Linux) and install it.
3. Troubleshooting
   If HDFView fails to launch or shows an error:
  - Install JDK 22, download link: https://www.oracle.com/java/technologies/javase/jdk22-archive-downloads.html
  - After installation, add the JDK bin directory to your system PATH environment variable.
  - Restart HDFView — it should now open correctly.

<img width="429" height="331" alt="1" src="https://github.com/user-attachments/assets/514dd6a1-aaeb-4c60-9814-e0eb30b8c12e" />

## Data Structure
The dataset is stored in HDF5 format and consists of three main components:
1. dynamic_data
  - Time-varying observations including temperature, relative humidity (RH), PM10, and NDVI.
  - The data are organized by timestamp, with each time step containing observation vectors from 6 stations (devices). This structure forms the “time × station × features” input sequence commonly used in spatio-temporal modeling.
2. static_data
    Station-level static attributes, including:
  - Sand barrier type
  - Instrument height above ground
  - Elevation
  - Latitude and longitude
3. metadata
    Descriptive information about the dataset.
