// Sentinel-2 + WorldCover export (Google Earth Engine)
// 1) Replace AOI with your WRF domain polygon (or import geometry)
// 2) Set date range (e.g., 2025-10)
// 3) Run and export to Drive

// AOI example: replace with your WRF domain
var aoi = ee.Geometry.Rectangle([115.633, 38.717, 116.333, 39.167]);

var start = '2025-10-01';
var end = '2025-10-31';

// Sentinel-2 L2A (surface reflectance)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(img){
    // scale reflectance
    var scl = img.select('SCL');
    var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
    img = img.updateMask(mask);
    var sr = img.select(['B2','B3','B4','B8','B11','B12']).multiply(0.0001);
    var ndvi = sr.normalizedDifference(['B8','B4']).rename('NDVI');
    var ndbi = sr.normalizedDifference(['B11','B8']).rename('NDBI');
    var ndmi = sr.normalizedDifference(['B8','B11']).rename('NDMI');
    return sr.addBands([ndvi, ndbi, ndmi]);
  });

// Median composite
var comp = s2.median().clip(aoi);

// ESA WorldCover 10m
var lc = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(aoi);

// Export NDVI/NDBI/NDMI
Export.image.toDrive({
  image: comp.select(['NDVI','NDBI','NDMI']),
  description: 'S2_indices_2025-10',
  folder: 'GEE_exports',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});

// Export land cover
Export.image.toDrive({
  image: lc,
  description: 'WorldCover_2021',
  folder: 'GEE_exports',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
