# Mapping nitrate contamination with geostatistics
### IDW and kriging interpolation, with greenhouse-proximity drift · Esposende – Vila do Conde Vulnerable Zone, Portugal

> Turning 500+ discrete well measurements into continuous nitrate surfaces
> for a Nitrates Directive Vulnerable Zone — comparing IDW and kriging
> variants under leave-one-out validation, quantifying uncertainty, and
> testing an external drift built from deep-learning-detected greenhouses.

---

## Why interpolate

Monitoring produces measurements at points; management needs answers about
*areas*. Spatial interpolation estimates the continuous distribution of
nitrate concentrations across the whole Vulnerable Zone from a discrete
set of sampling points, making it possible to identify pressure zones and
support decision-making under the Nitrates Directive (91/676/EEC) and its
**50 mg NO₃/L** quality limit.

But interpolation done casually is cartographic fiction. Every surface is
a model, every model has error, and the honest questions are: *which
method predicts best here, how large is the error, and where is the
estimate least trustworthy?* This project answers all three explicitly.

## The data

- **508 monitoring points** from the merger of two campaign layers
  (414 points, 2021–2024 · 94 points, 2025), unified with explicit field
  mapping;
- Pre-processing with documented rules: removal of null records, and
  **detection-limit handling** — values coded at the analytical limit
  replaced by 2.5 mg/L rather than silently kept or dropped;
- Concentrations spanning **2.5 to 376 mg/L** — a first hint of the
  extreme spatial heterogeneity to come.

## The approach

One script (`nitrate_interpolation.py`) runs the full comparison:

**1. IDW (p = 2), on raw values** — the deterministic baseline, run on
untransformed data (IDW is robust to the strong outliers of this series).

**2. Ordinary kriging, on log-transformed values** — competing variogram
models (spherical, exponential, gaussian) fitted and compared; the
log-transform tames the skewness that would otherwise distort variogram
fitting. Alongside every prediction surface, the **kriging variance** is
exported as an uncertainty map — the "where not to trust this map" layer.

**3. Universal kriging with an external drift — greenhouse proximity.**
The drift variable is built from the greenhouse layer produced by the
[U-Net detection project](https://github.com/LFilipePacheco/greenhouse-detection-unet):
a KD-tree computes, for every grid node, a proximity-weighted greenhouse
influence within a 3 km radius. The hypothesis under test: if intensive
horticulture drives contamination, adding its footprint as a trend should
improve prediction. One portfolio project feeding another — detection
output becoming a geostatistical covariate.

**4. Leave-one-out cross-validation** — every method judged by the same
objective standard: remove each point, predict it from the rest, score
RMSE and MAE over all 400+ usable points.

**5. GIS-ready outputs** — all surfaces and variance maps exported as
GeoTIFF (EPSG:3763), ready for ArcGIS/QGIS overlay with parcels, wells
and holdings.

## Results — and what they honestly say

| Method | RMSE (mg/L) | MAE (mg/L) |
|---|---|---|
| **IDW (p = 2)** | **41.2** | **28.8** |
| Kriging (exponential, log) | 43.3 | ~29 |
| Kriging (spherical, log) | 43.4 | 29.2 |

Three findings worth stating plainly:

- **The mean concentration (53.0 mg/L) exceeds the Directive limit** —
  the zone-wide picture is one of critical pressure, not isolated
  hotspots;
- **IDW edged out kriging, and the margin (~2 mg/L) is the message**:
  the spatial autocorrelation is too weak for kriging's model-based
  machinery to pay off. Contamination varies sharply over short
  distances — the signature of *local, practice-driven* pressure rather
  than smooth hydrogeological gradients;
- **The errors are large (RMSE ≈ 41 mg/L against a 2.5–376 range) and
  the report says so**: high spatial variability limits the precision of
  *any* interpolation here. The uncertainty maps are not an appendix —
  they are half the result.

## Beyond the maps: making the method teachable

Geostatistics enters institutions through people, not scripts. Alongside
the analysis, a **plain-language concepts guide** was written for
non-specialist colleagues — IDW vs kriging, what a semivariogram measures,
nugget/sill/range, why RMSE ≥ MAE always, and how to read a leave-one-out
validation — so the maps circulate together with the literacy to question
them.

## Repository contents

| File | Purpose |
|---|---|
| `nitrate_interpolation.py` | Full pipeline: ingestion, preprocessing, IDW, OK (multi-model), UK with greenhouse drift, LOO validation, GeoTIFF export |
| `docs/geostat_workflow.png` | Conceptual diagram: points → semivariogram → surface |
| `requirements.txt` | Python dependencies |

Paths are placeholders — point them at your own point layer, greenhouse
layer and output folder.

## Stack

Python · PyKrige · GeoPandas · SciPy (cKDTree) · rasterio · NumPy ·
matplotlib · GeoTIFF (EPSG:3763)

## About the data

The monitoring points and all derived surfaces are institutional data of
CCDR-Norte, I.P. and are not published here. Figures above are aggregate
validation statistics from the 2026 analysis. The diagram uses synthetic
data. The code is shared as a working reference implementation.

**Related projects:**
[nitrate monitoring pipeline](https://github.com/LFilipePacheco/monitorizacao-nitratos-zv) ·
[farm-holdings baseline](https://github.com/LFilipePacheco/farm-holdings-integration) ·
[greenhouse detection (U-Net)](https://github.com/LFilipePacheco/greenhouse-detection-unet) ·
[greenhouse registry verification](https://github.com/LFilipePacheco/greenhouse-registry-verification) ·
[livestock stocking rates](https://github.com/LFilipePacheco/livestock-stocking-rates)

---

**Luís Filipe Pacheco** — Senior Agricultural Engineer & Data Scientist,
CCDR-Norte, I.P. · [GitHub profile](https://github.com/LFilipePacheco) ·
[LinkedIn](https://www.linkedin.com/in/lu%C3%ADs-filipe-pacheco-471495b/) ·
[ORCID](https://orcid.org/0009-0001-7676-6542)
