# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                  matrix_product,
                                                  matrix_transpose)
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import (BaseCoordinateFrame,
                                           frame_transform_graph,
                                           RepresentationMapping)
from astropy.coordinates.attributes import (CoordinateAttribute,
                                            QuantityAttribute,
                                            DifferentialAttribute)
from astropy.coordinates.transformations import AffineTransform
from astropy.coordinates.errors import ConvertError

from astropy.coordinates import ICRS
from astropy.coordinates.builtin_frames.galactocentric import (Galactocentric,
                                                               _check_coord_repr_diff_types)

coord.galactocentric_frame_defaults.set('v4.0')

# The extra minus sign is because they report mu_W (west)
_m31_pmra = -(-125.2*u.km/u.s / (770*u.kpc)).to(u.mas/u.yr,
                                                u.dimensionless_angles())
_m31_pmdec = (-73.8*u.km/u.s / (770*u.kpc)).to(u.mas/u.yr,
                                               u.dimensionless_angles())

class LocalGroupBarycentric(BaseCoordinateFrame):
    r""" """
    frame_specific_representation_info = {
        r.CartesianDifferential: [
            RepresentationMapping('d_x', 'v_x', u.km/u.s),
            RepresentationMapping('d_y', 'v_y', u.km/u.s),
            RepresentationMapping('d_z', 'v_z', u.km/u.s),
        ],
    }

    default_representation = r.CartesianRepresentation
    default_differential = r.CartesianDifferential

    # frame attributes
    mw_coord = CoordinateAttribute(default=Galactocentric(x=0*u.kpc,
                                                          y=0*u.kpc,
                                                          z=0*u.kpc),
                                   frame=Galactocentric)

    # - distance from:
    #       https://ui.adsabs.harvard.edu/?#abs/2012ApJ...758...11C
    # - tangential velocity components from:
    #       https://ui.adsabs.harvard.edu/#abs/2012ApJ...753....8V/abstract
    # - heliocentric velocity from:
    #      https://ui.adsabs.harvard.edu/?#abs/2015ApJS..218...10V
    m31_coord = CoordinateAttribute(default=ICRS(ra=10.6847083*u.degree,
                                                 dec=41.26875*u.degree,
                                                 distance=779.*u.kpc,
                                                 pm_ra_cosdec=_m31_pmra,
                                                 pm_dec=_m31_pmdec,
                                                 radial_velocity=-358*u.km/u.s),
                                    frame=ICRS)

    mw_mass = QuantityAttribute(default=1E12*u.Msun)
    m31_mass = QuantityAttribute(default=2.5E12*u.Msun)


# Galactocentric to/from Local Group barycenter ----------------------->

def _check_coord_repr_diff_types(c, frame_name):
    if isinstance(c.data, r.UnitSphericalRepresentation):
        raise ConvertError("Transforming to/from a {0} frame "
                           "requires a 3D coordinate, e.g. (angle, angle, "
                           "distance) or (x, y, z).".format(frame_name))

    if ('s' in c.data.differentials and
            isinstance(c.data.differentials['s'],
                       (r.UnitSphericalDifferential,
                        r.UnitSphericalCosLatDifferential,
                        r.RadialDifferential))):
        raise ConvertError("Transforming to/from a {0} frame "
                           "requires a 3D velocity, e.g., proper motion "
                           "components and radial velocity.".format(frame_name))


def get_matrix_vectors(mw_cen, m31_cen, mw_mass, m31_mass, inverse=False):
    """
    Use the ``inverse`` argument to get the inverse transformation, matrix and
    offsets to go from LGBarycentric to Galactocentric.
    """

    m31_xyz = m31_cen.transform_to(mw_cen).cartesian.without_differentials()
    barycen = m31_xyz / (1 + mw_mass / m31_mass)
    barycen_sph = barycen.represent_as(r.SphericalRepresentation)

    # Align x(Galactocentric) with direction to M31
    mat1 = rotation_matrix(-barycen_sph.lat, 'y')
    mat2 = rotation_matrix(barycen_sph.lon, 'z')

    R = matrix_product(mat1, mat2)

    # Then we need to translate by the MW-barycenter distance
    offset = -r.CartesianRepresentation(barycen_sph.distance * [1., 0., 0.])

    if inverse:
        # the inverse of a rotation matrix is a transpose, which is much faster
        #   and more stable to compute
        R = matrix_transpose(R)
        offset = (-offset).transform(R)

    return R, offset


@frame_transform_graph.transform(AffineTransform, Galactocentric,
                                 LocalGroupBarycentric)
def galactocentric_to_lgbarycentric(galactocentric_coord, lgbarycentric_frame):
    _check_coord_repr_diff_types(galactocentric_coord, 'LocalGroupBarycentric')
    return get_matrix_vectors(lgbarycentric_frame.mw_coord,
                              lgbarycentric_frame.m31_coord,
                              lgbarycentric_frame.mw_mass,
                              lgbarycentric_frame.m31_mass)


@frame_transform_graph.transform(AffineTransform, LocalGroupBarycentric,
                                 Galactocentric)
def lgbarycentric_to_galactocentric(lgbarycentric_coord, galactocentric_frame):
    _check_coord_repr_diff_types(lgbarycentric_coord, 'LocalGroupBarycentric')
    return get_matrix_vectors(lgbarycentric_coord.mw_coord,
                              lgbarycentric_coord.m31_coord,
                              lgbarycentric_coord.mw_mass,
                              lgbarycentric_coord.m31_mass,
                              inverse=True)


@frame_transform_graph.transform(AffineTransform, LocalGroupBarycentric,
                                 LocalGroupBarycentric)
def lgbarycentric_to_lgbarycentric(c1, c2):
    _check_coord_repr_diff_types(c1, 'LocalGroupBarycentric')

    _, off1 = get_matrix_vectors(c1.mw_coord, c1.m31_coord,
                                 c1.mw_mass, c1.m31_mass)
    _, off2 = get_matrix_vectors(c2.mw_coord, c2.m31_coord,
                                 c2.mw_mass, c2.m31_mass)

    return np.eye(3), off2 - off1


# Config stuff:
MLG = 3.2e12 * u.Msun
mw_masses = np.array([0.9, 1.2, 1.5]) * 1e12*u.Msun
m31_masses = MLG - mw_masses

fiducial_mw_mass = 1.2e12 * u.Msun
fiducial_m31_mass = MLG - fiducial_mw_mass
