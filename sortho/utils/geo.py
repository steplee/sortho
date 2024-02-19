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
    wmLevels = np.array((
        20037508.342789248, 10018754.171394624, 5009377.085697312,  2504688.542848656,   1252344.271424328,
        626172.135712164,   313086.067856082,   156543.033928041,   78271.5169640205,    39135.75848201025,
        19567.879241005125, 9783.939620502562,  4891.969810251281,  2445.9849051256406,  1222.9924525628203,
        611.4962262814101,  305.7481131407051,  152.87405657035254, 76.43702828517627,   38.218514142588134,
        19.109257071294067, 9.554628535647034,  4.777314267823517,  2.3886571339117584,  1.1943285669558792,
        0.5971642834779396, 0.2985821417389698, 0.1492910708694849, 0.07464553543474245, 0.037322767717371225))

    @staticmethod
    def decimal2integral(x, lvl, cast=True):
        y =  (((x / Earth.WebMercatorScale) * .5 + .5) * (1<<lvl))
        if cast and isinstance(y,np.ndarray): y = y.astype(np.int64)
        elif cast: y = y.long()
        return y

    @staticmethod
    def integral2decimal(y, lvl, center=False):
        if isinstance(y,np.ndarray):
            if center:
                return ((y.astype(np.float64) + .5) / (1<<lvl) - .5) * 2 * Earth.WebMercatorScale
            else:
                return (y.astype(np.float64) / (1<<lvl) - .5) * 2 * Earth.WebMercatorScale
        else:
            if center:
                return ((y.double() + .5) / (1<<lvl) - .5) * 2 * Earth.WebMercatorScale
            else:
                return (y.double() / (1<<lvl) - .5) * 2 * Earth.WebMercatorScale

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
