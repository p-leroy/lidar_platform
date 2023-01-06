# coding: utf-8
# Baptiste Feldmann
# Paul Leroy
# SBET data handling

import copy
import math
import mmap
import os
import struct

import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import pyproj

from ..config.config import VERTICAL_DATUM_DIR

LIST_OF_ATTR = [('time', 'float64'),
                ('latitude', 'float64'),
                ('longitude', 'float64'),
                ('elevation', 'float32'),
                ('x_vel', 'float32'),
                ('y_vel', 'float32'),
                ('z_vel', 'float32'),
                ('roll', 'float32'),
                ('pitch', 'float32'),
                ('heading', 'float32'),
                ('wander_angle', 'float32'),
                ('x_acceleration', 'float32'),
                ('y_acceleration', 'float32'),
                ('z_acceleration', 'float32'),
                ('x_angular_rate', 'float32'),
                ('y_angular_rate', 'float32'),
                ('z_angular_rate', 'float32')]

field_names = ('time (s)',
               'latitude (deg)',
               'longitude (deg)',
               'hauteur (m)',
               'x_vel (m/s)',
               'y_vel (m/s)',
               'z_vel (m/s)',
               'roll (rad)',
               'pitch (rad)',
               'platform_heading (rad)',
               'wander_angle (rad)',
               'x_acceleration (m/s²)',
               'y_acceleration (m/s²)',
               'z_acceleration (m/s²)',
               'x_angular_rate (rad/s)',
               'y_angular_rate (rad/s)',
               'z_angular (rad/s)')

# 17 attributes of 8 bytes each = 136 bytes
LINE_SIZE = int(136)


def merge_sbet(sbet_list):
    new = copy.deepcopy(sbet_list[0])

    for i in range(1, len(sbet_list)):
        new.array = np.append(new.array, sbet_list[i].array)

    new.gps_time = new.array['time']
    new.latitude = new.array['latitude']
    new.longitude = new.array['longitude']
    new.elevation = new.array['elevation']

    return new


class SBET(object):
    def __init__(self, filepath):
        self.filepath = filepath

        # load trajectory data
        self.array = None
        self.gps_time = None
        self.latitude = None
        self.longitude = None
        self.elevation = None
        self.load_data()

        self.easting = None
        self.northing = None
    
    def __str__(self):
        return "\n".join(self.__dict__.keys())
    
    def load_data(self):
        if os.path.splitext(self.filepath)[-1] != ".out":
            raise OSError("Unknown file extension, can read only .out file !")
        else:
            print(f'[SBET.load_data] load data from {self.filepath}')

        f = open(self.filepath, mode='rb')
        f_size = os.path.getsize(self.filepath)
        data = mmap.mmap(f.fileno(), f_size, access=mmap.ACCESS_READ)
        nbr_line = int(len(data) / LINE_SIZE)
        print(f'[SBET.load_data] number of lines in SBET files: {nbr_line}')

        temp = []
        for i in range(0, nbr_line):
            temp += [struct.unpack('17d', data[i * LINE_SIZE:(i + 1) * LINE_SIZE])]

        self.array = np.array(temp, dtype=LIST_OF_ATTR)
        self.array['latitude'] *= 180 / math.pi
        self.array['longitude'] *= 180 / math.pi

        self.gps_time = self.array['time']
        self.latitude = self.array['latitude']
        self.longitude = self.array['longitude']
        self.elevation = self.array['elevation']
    
    def _compute_undulation(self, geoid_grid):
        # npzfile=np.load("G:/RENNES1/BaptisteFeldmann/Vertical_datum/"+geoidgrid+".npz")
        npz = np.load(VERTICAL_DATUM_DIR + geoid_grid + ".npz")
        grille = npz[npz.files[0]]
        undulation = griddata(grille[:, 0: 2],grille[:, 2], (self.longitude, self.latitude), method='linear')
        return undulation

    def h2he(self, geoid_grid):
        # ellipsoidal height to altitude
        undulation = self._compute_undulation(geoid_grid)
        altitude = self.elevation - undulation
        self.array['elevation'] = altitude
        self.elevation = self.array['elevation']
    
    def he2h(self, geoid_grid):
        # altitude to ellipsoidal height
        undulation = self._compute_undulation(geoid_grid)
        height = self.elevation + undulation
        self.array['elevation'] = height
        self.elevation = self.array['elevation']

    def projection(self, epsg_in, epsg_out):
        # Conversion geo coordinates tp projected coordinates
        # epsg:4171 -> ETRS89-géo lat,long,h
        # epsg:2154 -> RGF93-L93 E,N,H
        # epsg:4326 -> WGS84-géo lat,long,h
        # epsg:32634 -> WGS84-UTM 34N E,N,H
        # epsg:4167 -> NZGD2000 lat,long,h
        # epsg:2193 -> NZTM2000 E,N,H
        transformer = pyproj.Transformer.from_crs(epsg_in, epsg_out)
        self.easting, self.northing = transformer.transform(self.latitude, self.longitude)
    
    def export(self, epsg_in, epsg_out):
        if not hasattr(self, "easting"):
            self.projection(epsg_in, epsg_out)

        data = np.array([self.easting, self.northing, self.elevation, self.gps_time])
        out = self.filepath[0:-4] + "_ascii.txt"
        f = np.savetxt(out, np.transpose(data),
                       fmt="%.3f;%.3f;%.3f;%f", delimiter=";", header="X;Y;Z;gpstime")
        return out

    def interpolate(self, time_ref):
        temp = np.transpose([self.easting, self.northing, self.elevation])
        f = interp1d(self.gps_time, temp, axis=0)
        return f(time_ref)


def calc_grid(name_geoid, pts0, deltas):
    """
    Function for compute a geoid grid from an Ascii file.
    To use a geoid grid, you have to register a NPZ file
    with a specific format. To formalize the geoid grid file,
    use this function.

    Parameters
    ----------
    name_geoid : str
            name of the Ascii file that you want to formalize
            ex : "RAF09"
    pts0 : ndarray
           geographical coordinates of the top left point
    deltas : ndarray
           steps sampling between grid points

    Return
    ------
    Register a formalize geoid grid in NPZ file
    """
    npzfile = np.load("D:/TRAVAIL/Vertical_datum/" + name_geoid + "_brut.npz")
    grille = npzfile[npzfile.files[0]]
    taille = np.shape(grille)
    tableau = np.zeros((taille[0]*taille[1],3))
    compteur = 0
    for lig in range(0, taille[0]):
        for col in range(0, taille[1]):
            tableau[compteur, 0] = round(pts0[0]+deltas[0] * col, 6)
            tableau[compteur, 1] = round(pts0[1]-deltas[1] * lig, 6)
            tableau[compteur, 2] = grille[lig, col]
            compteur += 1
    np.savez_compressed("D:/TRAVAIL/Vertical_datum/RAF09.npz", tableau)
    return True


def Projection(epsg_in, epsg_out, x, y, z):
    """
    Function for compute the transformation between
    2 references system. This function use PyProj library.

    Parameters
    ----------
    epsg_in : int
              EPSG code of source reference system
              ex : 4171
    epsg_out : int
               EPSG code of target reference system
               ex : 2154
    x : ndarray
        Abscissa axis data.
        For geographical coordinates it's latitude
        For projection coordinates it's Easting
    y : ndarray
        Ordinate axis data.
        For geographical coordinates it's longitude
        For projection coordinates it's Northing
    z : ndarray
        Elevation data
        This field is required but not modified
        
    Return
    ------
    result : ndarray
            table of the transform data
    """
    transformer = pyproj.Transformer.from_crs(epsg_in, epsg_out)
    temp = transformer.transform(x, y, z)
    result = np.array([temp[0], temp[1], temp[2]])
    return np.transpose(result)


def sbet_config(filepath):
    sbet_dict = {}
    for i in np.loadtxt(filepath, str, delimiter="="):
        sbet_dict[i[0]] = i[1]
    
    sbet_list = []
    for i in sbet_dict['listFiles'].split(','):
        sbet_file = os.path.join(sbet_dict['path'], str(i))
        sbet_list += [SBET(sbet_file)]
    
    if len(sbet_list) > 1:
        sbet_obj = merge_sbet(sbet_list)
    else:
        sbet_obj = sbet_list[0]
    
    if sbet_dict['Z'] == 'height':
        sbet_obj.h2he(sbet_dict['geoidgrid'])

    sbet_obj.projection(int(sbet_dict['epsg_source']), int(sbet_dict['epsg_target']))

    return sbet_obj
