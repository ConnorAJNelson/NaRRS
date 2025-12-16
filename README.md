## Sentinel-3 NaRRS: Nadir Radiometer and Radar Synergy for Arctic Snow Depth and Sea Ice Thickness Retrieval ##

Code repository to accompany the manuscript "Snow depth on Arctic sea ice retrieval using a synergy of Sentinel-3's active and passive microwave instruments". The correspondong data product can be found at https://doi.org/10.5281/zenodo.17942492 and https://www.cpom.ucl.ac.uk/narss/.

Repository DOI (v1.0.0-alpha): 10.5281/zenodo.17955873

<img width="1582" height="1087" alt="NaRRS_ql_20220414" src="https://github.com/user-attachments/assets/c6dfa1ae-a825-4de8-acd6-03609735d10d" />
Fig: Example NaRRS snow depth and sea ice thickness for a single date of along-track Sentinel-3 observations, and shown zoomed-in over a coincident OLCI true color composite. 

### Data Sources ###

* Sentinel-3 Level 2 Sea Ice Thematic Product (Aublanc et al., 2025) which contains radiometer (MWR) and altimeter (SRAL) measurements. The product can be downloaded from the Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/), see https://doi.org/10.57780/s3d-6c5ea43 for details.
* Sea ice concentration (OSI-430-a) https://osi-saf.eumetsat.int/products/osi-430-a
* Sea ice concentration (OSI-450-a) https://osi-saf.eumetsat.int/products/osi-450-a
* Sea ice type (OSI-403-d) https://osi-saf.eumetsat.int/products/osi-403-d
* Sea ice drift vectors (OSI-405-c) https://osi-saf.eumetsat.int/products/osi-405-c
* Sea ice drift vectors (OSI-455) https://osi-saf.eumetsat.int/products/osi-455
* ERA5 surface and atmospheric analysis-ready reanalysis https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5
* TELSEM<sup>2</sup> emissivity atlas https://nwp-saf.eumetsat.int/site/software/rttov/download/
* NSIDC sea ice region mask https://nsidc.org/data/nsidc-0780/versions/1
* Operation IceBridge snow depth https://nsidc.org/data/nsidc-0708/versions/1
* AWI IceBird snow depth, sea ice thickness etc https://doi.pangaea.de/10.1594/PANGAEA.966009 and https://doi.pangaea.de/10.1594/PANGAEA.966057
* AWI snow buoy snow depths https://data.meereisportal.de/relaunch/buoy.php
* Beaufort Gyre Exploraiton Project (BGEP) sea ice draft https://www2.whoi.edu/site/beaufortgyre/data/
* Fram Strait Arctic Outflow Observatory (AOO) sea ice draft  https://data.npolar.no/dataset/5b717274-2d85-4f13-a1b4-ff0517c78b4a

### Required Software ###
* RTTOV (for atmospheric correction of brightness temperatures) https://nwp-saf.eumetsat.int/site/software/rttov/download/
* pysiral (for altimetric waveform parameter retrieval) https://github.com/pysiral/pysiral


References:
Aublanc, J., Renou, J., Piras, F. et al. Sentinel-3 Altimetry Thematic Products for Hydrology, Sea Ice and Land Ice. Sci Data 12, 714 (2025). https://doi.org/10.1038/s41597-025-04956-3
