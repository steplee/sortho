import numpy as np

# WARNING: Wow they really swapped lat/lng in GDAL 3.0
#          We need to check version after importing and account for it.
#          I always want lng-lat-alt order.
try:
    import osr
    import gdal
    MUST_SWAP_LNG_LAT = int(gdal.__version__[0]) >= 3
except:
    import osgeo
    import osgeo.osr as osr
    MUST_SWAP_LNG_LAT = int(osgeo.__version__[0]) >= 3


class Earth:
    R1         = (6378137.0);
    R2         = (6356752.314245179);
    R1_inv     = (1. / 6378137.0);
    WGS84_F    = (1. / 298.257223563);
    WGS84_D    = (R2 / R1);
    a          = 1;
    b          = R2 / R1;
    a2         = a * a;
    b2         = b * b;
    e2         = 1 - (b * b / a * a);
    ae2        = a * e2;
    b2_over_a2 = b2 / a2;
    one_div_pi = 1 / np.pi;

    WebMercatorScale = 20037508.342789248;

def transform_points_epsg(from_, to_, p):
    # import time
    # t = time.time()
    if MUST_SWAP_LNG_LAT and from_ == 4326:
        p = p[...,[1,0,2]]
    p = np.array(get_converter_from_epsg_(from_,to_).TransformPoints(p))
    if MUST_SWAP_LNG_LAT and to_ == 4326:
        p = p[...,[1,0,2]]
    # print(f'xform took {(time.time()-t) * 1000:.1f}ms')
    return p

def tlbr2corners(tlbr):
    tlbr = tlbr.reshape(-1)
    return np.array((
        tlbr[0], tlbr[1],
        tlbr[2], tlbr[1],
        tlbr[2], tlbr[3],
        tlbr[0], tlbr[3])).reshape(4,2)

epsg_map_ = {}
def get_converter_from_epsg_(src_, dst_):
    global epsg_map_
    k = str(src_) + str(dst_)
    if k in epsg_map_:
        return epsg_map_[k]
    src = osr.SpatialReference()
    dst = osr.SpatialReference()
    src.ImportFromEPSG(src_)
    dst.ImportFromEPSG(dst_)
    ct = osr.CoordinateTransformation(src,dst)
    epsg_map_[k] = ct
    return ct
def get_converter_from_wkt_(srcWkt_, dstEpsg_):
    src = osr.SpatialReference()
    dst = osr.SpatialReference()
    src.ImportFromWkt(srcWkt_)
    dst.ImportFromEPSG(dstEpsg_)
    ct = osr.CoordinateTransformation(src,dst)
    return ct
def get_converter_to_wkt_(srcEpsg_, dstWkt_):
    src = osr.SpatialReference()
    dst = osr.SpatialReference()
    src.ImportFromEPSG(srcEpsg_)
    dst.ImportFromWkt(dstWkt_)
    ct = osr.CoordinateTransformation(src,dst)
    return ct
