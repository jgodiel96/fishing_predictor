# Oceanographic Data Sources for Fishing Prediction

This document provides a comprehensive list of free/open APIs and data sources for oceanographic data that can be used for fishing prediction, with special focus on South America/Peru coverage.

---

## 1. Sea Surface Temperature (SST) Data Sources

### 1.1 NOAA OISST (Optimum Interpolation SST)

| Attribute | Details |
|-----------|---------|
| **Provider** | NOAA NCEI |
| **Dataset** | NOAA 1/4° Daily Optimum Interpolation SST v2.1 |
| **Resolution** | 0.25° x 0.25° (~25km) |
| **Temporal** | Daily, from September 1981 to present |
| **Coverage** | Global |
| **Format** | NetCDF |
| **Cost** | Free, no account needed |
| **Update Frequency** | Daily (1-2 day lag) |

**Access Methods:**
- **Direct HTTPS**: `https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/`
- **ERDDAP**: Available on multiple ERDDAP servers
- **Google Earth Engine**: `ee.ImageCollection("NOAA/CDR/OISST/V2_1")`

**Python Example:**
```python
import xarray as xr

# Direct access via OPeNDAP
url = "https://www.ncei.noaa.gov/thredds/dodsC/OisshiresDaily"
ds = xr.open_dataset(url)
```

**Documentation**: https://www.ncei.noaa.gov/products/optimum-interpolation-sst

---

### 1.2 NOAA Geo-Polar Blended SST (GPB)

| Attribute | Details |
|-----------|---------|
| **Provider** | NOAA OSPO |
| **Resolution** | ~5km (0.05°) |
| **Temporal** | 3 daily products (night-only, day-night, diurnally corrected) |
| **Coverage** | Global |
| **Format** | NetCDF4 (GHRSST GDS2 compliant) |
| **Cost** | Free |
| **Latency** | Near real-time |

**Access**: https://www.ospo.noaa.gov/products/ocean/sst.html

---

### 1.3 NASA PO.DAAC - GHRSST MUR SST

| Attribute | Details |
|-----------|---------|
| **Provider** | NASA JPL PO.DAAC |
| **Dataset** | Multi-scale Ultra-high Resolution (MUR) SST |
| **Resolution** | 1km (0.01°) |
| **Temporal** | Daily |
| **Coverage** | Global |
| **Format** | NetCDF4 |
| **Cost** | Free (requires Earthdata login) |

**Access Methods:**
- **Earthdata Search**: https://search.earthdata.nasa.gov
- **podaac-data-subscriber** (recommended for bulk downloads):
```bash
pip install podaac-data-subscriber
podaac-data-downloader -c MUR-JPL-L4-GLOB-v4.1 -d ./data --start-date 2024-01-01T00:00:00Z --end-date 2024-01-31T00:00:00Z
```

**Documentation**: https://podaac.jpl.nasa.gov/GHRSST

---

### 1.4 Copernicus Marine Service - SST Products

| Attribute | Details |
|-----------|---------|
| **Provider** | Copernicus Marine Service (CMEMS) |
| **Products** | Multiple global and regional SST products |
| **Resolution** | Various (0.05° to 0.25°) |
| **Temporal** | Daily, near real-time and reanalysis |
| **Coverage** | Global |
| **Format** | NetCDF, Zarr |
| **Cost** | Free (requires registration) |

**Python Access (copernicusmarine toolbox):**
```python
pip install copernicusmarine

import copernicusmarine

# Login (first time)
copernicusmarine.login()

# Download SST data
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    variables=["thetao"],
    minimum_longitude=-90,
    maximum_longitude=-70,
    minimum_latitude=-20,
    maximum_latitude=0,
    start_datetime="2024-01-01",
    end_datetime="2024-01-31",
    output_filename="peru_sst.nc"
)
```

**Registration**: https://data.marine.copernicus.eu/register
**Documentation**: https://help.marine.copernicus.eu/

---

## 2. Ocean Currents Data

### 2.1 HYCOM (Hybrid Coordinate Ocean Model)

| Attribute | Details |
|-----------|---------|
| **Provider** | HYCOM Consortium / NOAA NCEP |
| **Resolution** | 1/12° (~8km) at equator |
| **Temporal** | Daily forecasts, hourly available |
| **Depth Levels** | 40 standard z-levels |
| **Coverage** | Global |
| **Format** | NetCDF |
| **Cost** | Free |
| **Variables** | Eastward/Northward velocity, temperature, salinity |

**Access Methods:**

1. **NOAA ERDDAP**:
```
https://www.ncei.noaa.gov/erddap/griddap/Hycom_sfc_3d.html
```

2. **Google Earth Engine**:
```python
# HYCOM sea water velocity
ee.ImageCollection("HYCOM/sea_water_velocity")
```

3. **NOMADS Server** (operational forecasts):
```
http://nomads.ncep.noaa.gov/
```

**Documentation**: https://www.hycom.org/dataserver

---

### 2.2 Copernicus Marine - MERCATOR Global Ocean Model

| Attribute | Details |
|-----------|---------|
| **Provider** | Mercator Ocean International / CMEMS |
| **Product ID** | GLOBAL_ANALYSISFORECAST_PHY_001_024 |
| **Resolution** | 1/12° (~8km) |
| **Temporal** | Hourly, 6-hourly, Daily, Monthly |
| **Depth Levels** | 50 levels (0-5500m) |
| **Coverage** | Global |
| **Format** | NetCDF, Zarr |
| **Cost** | Free (requires registration) |
| **Forecast** | 10 days ahead, updated daily |

**Available Current Datasets:**
- `cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i` - 6-hourly currents
- `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` - Daily mean currents
- `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` - Hourly surface currents (includes wave/tidal drift)

**Python Access:**
```python
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["uo", "vo"],  # eastward and northward velocity
    minimum_longitude=-90,
    maximum_longitude=-70,
    minimum_latitude=-20,
    maximum_latitude=0,
    start_datetime="2024-01-01",
    end_datetime="2024-01-31",
    output_filename="peru_currents.nc"
)
```

**Data Portal**: https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/

---

### 2.3 OSCAR (Ocean Surface Current Analysis Real-time)

| Attribute | Details |
|-----------|---------|
| **Provider** | NOAA / ESR |
| **Resolution** | 1/3° (~33km) |
| **Temporal** | 5-day averages |
| **Coverage** | Global (60°S to 60°N) |
| **Format** | NetCDF |
| **Cost** | Free |

**Access**: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg

---

## 3. Chlorophyll-a / Ocean Productivity Data

### 3.1 NASA OB.DAAC - MODIS Chlorophyll-a

| Attribute | Details |
|-----------|---------|
| **Provider** | NASA Ocean Biology DAAC |
| **Sensors** | MODIS-Aqua, MODIS-Terra |
| **Resolution** | 4km (Level 3 SMI) |
| **Temporal** | Daily, 8-day, Monthly |
| **Coverage** | Global |
| **Period** | 2002-present (Aqua), 2000-present (Terra) |
| **Format** | NetCDF, HDF |
| **Cost** | Free (Earthdata login required) |

**Access Methods:**

1. **Direct HTTPS**:
```
https://oceandata.sci.gsfc.nasa.gov/
```

2. **Python (earthaccess)**:
```python
pip install earthaccess

import earthaccess

earthaccess.login()
results = earthaccess.search_data(
    short_name="MODISA_L3m_CHL",
    temporal=("2024-01-01", "2024-01-31"),
    bounding_box=(-90, -20, -70, 0)  # Peru region
)
earthaccess.download(results, "./data/")
```

3. **Google Earth Engine**:
```python
# MODIS Aqua Ocean Color
ee.ImageCollection("NASA/OCEANDATA/MODIS-Aqua/L3SMI")
```

**Documentation**: https://oceancolor.gsfc.nasa.gov/

---

### 3.2 Sentinel-3 OLCI Chlorophyll-a

| Attribute | Details |
|-----------|---------|
| **Provider** | ESA / EUMETSAT / Copernicus |
| **Sensor** | OLCI (Ocean and Land Colour Instrument) |
| **Resolution** | 300m |
| **Temporal** | ~2 days global coverage |
| **Coverage** | Global |
| **Format** | NetCDF |
| **Cost** | Free |

**Access Methods:**

1. **Copernicus Data Space Ecosystem**:
```
https://dataspace.copernicus.eu/
```

2. **Google Earth Engine**:
```python
ee.ImageCollection("COPERNICUS/S3/OLCI")
```

3. **NOAA CoastWatch**:
```
https://coastwatch.noaa.gov/cwn/products/ocean-color-near-real-time-olci-sentinel-3a-and-3b-global-coverage.html
```

---

### 3.3 VIIRS Chlorophyll-a

| Attribute | Details |
|-----------|---------|
| **Provider** | NOAA / NASA |
| **Sensors** | VIIRS (Suomi-NPP, NOAA-20, NOAA-21) |
| **Resolution** | 750m (Level 2), 4km (Level 3) |
| **Temporal** | Daily |
| **Coverage** | Global |
| **Format** | NetCDF |
| **Cost** | Free |

**Access**:
- NOAA CoastWatch: https://coastwatch.noaa.gov/
- NASA OB.DAAC: https://oceancolor.gsfc.nasa.gov/

---

### 3.4 Copernicus Marine - Ocean Color Products

| Attribute | Details |
|-----------|---------|
| **Provider** | CMEMS |
| **Products** | Multiple L3/L4 chlorophyll products |
| **Resolution** | 4km |
| **Temporal** | Daily, Monthly |
| **Coverage** | Global |
| **Format** | NetCDF |
| **Cost** | Free (registration required) |

**Product IDs:**
- `OCEANCOLOUR_GLO_BGC_L4_MY_009_104` - Multi-year reprocessed
- `OCEANCOLOUR_GLO_BGC_L4_NRT_009_102` - Near real-time

---

## 4. Fishing Vessel and Activity Data

### 4.1 Global Fishing Watch API

| Attribute | Details |
|-----------|---------|
| **Provider** | Global Fishing Watch |
| **Data Types** | Fishing effort, vessel presence, encounters, port visits |
| **Coverage** | Global (2012-present) |
| **Format** | JSON, GeoJSON |
| **Cost** | Free for non-commercial use |
| **Registration** | Required (API key) |

**Available APIs:**
- **4Wings API**: Fishing effort, vessel presence, SAR detections (2017-present)
- **Vessel API**: Vessel identity and search
- **Events API**: Encounters, loitering, port visits, fishing events
- **Insights API**: Risk assessment and IUU indicators
- **Bulk Download API**: Large volume data access

**Python Package:**
```python
pip install gfw-api-python-client

# Or use the R package
# install.packages("gfwr")
```

**API Registration**: https://globalfishingwatch.org/our-apis/
**Documentation**: https://globalfishingwatch.org/our-apis/documentation

**Peru-specific**: Global Fishing Watch has a partnership with Peru, monitoring ~2,000 vessels (industrial and artisanal) since 2018.

---

### 4.2 AIS Data Sources

#### 4.2.1 AISHub (Free Data Exchange)

| Attribute | Details |
|-----------|---------|
| **Provider** | AISHub |
| **Coverage** | Global (community-contributed) |
| **Format** | JSON, XML, CSV |
| **Cost** | Free (data exchange model) |
| **Requirement** | Must contribute AIS data to receive access |

**Access**: https://www.aishub.net/

#### 4.2.2 aisstream.io

| Attribute | Details |
|-----------|---------|
| **Provider** | aisstream.io |
| **Coverage** | Global |
| **Format** | JSON (WebSocket API) |
| **Cost** | Free tier available |
| **Type** | Real-time streaming |

**Access**: https://aisstream.io/

#### 4.2.3 NOAA AccessAIS (US Waters)

| Attribute | Details |
|-----------|---------|
| **Provider** | NOAA |
| **Coverage** | US coastal waters |
| **Format** | Various (interactive download) |
| **Cost** | Free |
| **Update** | Quarterly |

**Access**: https://www.coast.noaa.gov/digitalcoast/tools/ais.html

---

### 4.3 FAO Fisheries Statistics

| Attribute | Details |
|-----------|---------|
| **Provider** | FAO |
| **Data Types** | Capture production, aquaculture, trade |
| **Coverage** | Global (200+ countries) |
| **Period** | 1976-present |
| **Format** | CSV, through FishStatJ software |
| **Cost** | Free |
| **Update** | Annual (Q2 typically) |

**Access Methods:**

1. **FishStatJ** (Desktop application):
   - Download: https://www.fao.org/fishery/statistics/software/fishstatj/en

2. **Online Query Panel**:
   - https://www.fao.org/fishery/statistics-query/

3. **R Package (fishstat)**:
```r
install.packages("fishstat")
library(fishstat)
```

**Data Portal**: https://www.fao.org/fishery/statistics/en

---

## 5. ERDDAP - Universal Data Access

ERDDAP servers provide a unified interface to access multiple oceanographic datasets.

### Major ERDDAP Servers

| Server | URL | Specialty |
|--------|-----|-----------|
| **NOAA CoastWatch** | https://coastwatch.pfeg.noaa.gov/erddap/ | US coastal data |
| **NOAA NCEI** | https://www.ncei.noaa.gov/erddap/ | Climate data records |
| **IOOS** | https://erddap.ioos.us/erddap/ | US ocean observing |
| **CMEMS** | Via OPeNDAP | European marine data |

### Python Client (erddapy)
```python
pip install erddapy

from erddapy import ERDDAP

server = ERDDAP(
    server="https://coastwatch.pfeg.noaa.gov/erddap/",
    protocol="griddap"
)

# Search for SST datasets
server.dataset_id = "ncdcOisst21Agg_LonPM180"
server.variables = ["sst", "anom"]
server.constraints = {
    "time>=": "2024-01-01",
    "time<=": "2024-01-31",
    "latitude>=": -20,
    "latitude<=": 0,
    "longitude>=": -90,
    "longitude<=": -70
}

# Get data as xarray Dataset
ds = server.to_xarray()
```

**Search NOAA Datasets**: https://noaa-erddap.org/

---

## 6. Peru/South America Specific Considerations

### Coverage Assessment

| Data Source | Peru/Humboldt Coverage | Notes |
|-------------|----------------------|-------|
| NOAA OISST | Excellent | Global daily coverage |
| GHRSST MUR | Excellent | 1km resolution ideal for coastal areas |
| CMEMS MERCATOR | Excellent | Strong coverage of Peru Current |
| HYCOM | Excellent | Good resolution for Humboldt system |
| MODIS Chl-a | Good | Cloud issues in coastal Peru |
| Sentinel-3 OLCI | Good | 300m ideal for coastal productivity |
| Global Fishing Watch | Excellent | Direct partnership with Peru since 2018 |

### IMARPE (Instituto del Mar del Perú)

Peru's national marine research institute (IMARPE) conducts extensive monitoring of the Humboldt Current system:
- **Data collected**: SST, salinity, currents, chlorophyll, fish biomass
- **Coverage**: Peruvian waters to 200nm offshore
- **Frequency**: 4-5 ship surveys per year
- **Note**: No public API available; data may be accessible through direct contact or international partnerships

**Contact**: https://www.imarpe.gob.pe/

---

## 7. Recommended Data Stack for Fishing Prediction

### Primary Sources (Recommended)

| Variable | Source | Reason |
|----------|--------|--------|
| SST | NOAA OISST | Long history, reliable, free |
| SST (high-res) | GHRSST MUR | 1km for coastal detail |
| Currents | CMEMS MERCATOR | High resolution, forecasts |
| Chlorophyll-a | NASA MODIS/OB.DAAC | Long time series |
| Chlorophyll (high-res) | Sentinel-3 OLCI | 300m coastal detail |
| Fishing Activity | Global Fishing Watch | Peru partnership, free API |
| Historical Fish Data | FAO FishStat | Long-term trends |

### Python Dependencies
```
copernicusmarine
earthaccess
erddapy
xarray
netCDF4
gfw-api-python-client
fishstat (R)
```

---

## 8. Quick Reference: API Endpoints

| Service | Endpoint/Method |
|---------|----------------|
| NOAA OISST | `https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/` |
| NASA PO.DAAC | `podaac-data-downloader` CLI tool |
| NASA OB.DAAC | `earthaccess` Python library |
| CMEMS | `copernicusmarine` Python library |
| ERDDAP | `erddapy` Python library |
| Global Fishing Watch | `https://gateway.api.globalfishingwatch.org/` |
| FAO | FishStatJ desktop app or `fishstat` R package |

---

## Sources

### SST Data
- [NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- [NASA PO.DAAC](https://podaac.jpl.nasa.gov/)
- [NOAA OSPO SST Products](https://www.ospo.noaa.gov/products/ocean/sst.html)
- [NOAA CoastWatch SST](https://coastwatch.noaa.gov/cwn/product-families/sea-surface-temperature.html)

### Ocean Currents
- [HYCOM Data Server](https://www.hycom.org/dataserver)
- [CMEMS Global Ocean Physics](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services)
- [Google Earth Engine HYCOM](https://developers.google.com/earth-engine/datasets/catalog/HYCOM_sea_water_velocity)

### Chlorophyll/Ocean Color
- [NASA OB.DAAC](https://oceancolor.gsfc.nasa.gov/)
- [NASA Earthdata](https://www.earthdata.nasa.gov/centers/ob-daac)
- [Sentinel-3 OLCI on GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI)
- [NOAA CoastWatch Ocean Color](https://coastwatch.noaa.gov/cwn/products/ocean-color-near-real-time-olci-sentinel-3a-and-3b-global-coverage.html)

### Fishing/Vessel Data
- [Global Fishing Watch APIs](https://globalfishingwatch.org/our-apis/)
- [Global Fishing Watch Documentation](https://globalfishingwatch.org/our-apis/documentation)
- [AISHub](https://www.aishub.net/)
- [aisstream.io](https://aisstream.io/)
- [FAO Fisheries Statistics](https://www.fao.org/fishery/statistics/en)

### General Access
- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [CMEMS Registration](https://data.marine.copernicus.eu/register)
- [ERDDAP Documentation](https://erddap.github.io/)
- [NOAA ERDDAP Search](https://noaa-erddap.org/)

---

## 9. Integration with Project

### 9.1 Using Domain Constants

All API endpoints and dataset configurations are centralized in `domain.py`:

```python
from domain import ENDPOINTS, ERDDAP_DATASETS, STUDY_AREA

# API endpoints
print(ENDPOINTS.erddap_base)      # https://coastwatch.pfeg.noaa.gov/erddap
print(ENDPOINTS.openmeteo_marine) # https://marine-api.open-meteo.com/v1/marine

# ERDDAP datasets
for ds in ERDDAP_DATASETS:
    print(f"{ds.dataset_id}: {ds.description}")

# Study area bounds
print(f"Region: {STUDY_AREA.north}N to {STUDY_AREA.south}S")
```

### 9.2 Data Fetcher Usage

```python
from core.marine_data import MarineDataFetcher
from data.fetchers.historical_fetcher import HistoricalDataFetcher

# Real-time data
fetcher = MarineDataFetcher()
points = fetcher.fetch_points(coordinates)

# Historical data (requires API keys)
hist_fetcher = HistoricalDataFetcher()
sst_data = hist_fetcher.fetch_noaa_erddap_sst("2025-01-01", "2026-01-28")
```

### 9.3 No Synthetic Data Policy

This project uses **ONLY real data**. All data sources listed in this document provide actual satellite observations or validated reanalysis products. See `DEVELOPMENT_GUIDELINES.md` for data handling policies.

---

*Document updated: 2026-01-28*
*For fishing prediction project - Peru/South America focus*
*See also: `DEVELOPMENT_GUIDELINES.md` for coding standards*
