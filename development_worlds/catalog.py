"""Public development domains and validation cases, design revision 1.

Cases are separate from first light and from an agent's acquired observations.
These six public suites are for development, not the official leaderboard.
"""
from __future__ import annotations
import copy
import numpy as np

WORLDS = ('1', '2', '3', '4', '5', '6')
SCALAR_WORLDS = frozenset(('1', '2', '4'))
VERSION = 'public-development-design-1.0.0'
# r_min, r_max, last absolute time, maximum trajectory duration, speed limit
LIMITS = {
    '1': (1., 8., 40., 30., 5.),
    '2': (.5, 6., 32., 24., 3.),
    '3': (.5, 6., 32., 24., 3.),
    '4': (1., 10., 40., 30., 5.),
    '5': (.5, 5., 32., 20., 3.),
    '6': (.5, 6., 48., 24., 3.),
}

def _world(world):
    value = str(world)
    if value not in WORLDS:
        raise ValueError('Development world must be 1 through 6.')
    return value

def _probe(world, pos, vel, attr=1.):
    return {'pos': list(map(float, pos)), 'vel': list(map(float, vel)),
            'properties': {'p2' if world in SCALAR_WORLDS else 'mass': float(attr)}}

def first_light(world):
    world = _world(world)
    positions = [[2., 0.], [0., 4.]] if world in ('1', '4') else [[1.5, 0.], [0., 3.]]
    return {'t_launch': 0., 'readout': [0., .05, .1, .2, .35, .5, .7, 1.],
            'source_properties': {'p1': 1.} if world in SCALAR_WORLDS else {},
            'probes': [_probe(world, p, v) for p, v in
                       zip(positions, [[0., .3], [-.3, 0.]])]}

def config(world):
    world = _world(world)
    rmin, rmax, tmax, duration, vmax = LIMITS[world]
    return {'world_id': world, 'status': 'design_prototype_not_frozen',
            'catalog_version': VERSION,
            'launch_region': {'kind': 'annulus', 'center': [0., 0.],
                              'radius_min': rmin, 'radius_max': rmax},
            'initial_speed_max': vmax, 'maximum_physical_time': tmax,
            'maximum_single_trajectory_duration': duration,
            'source_properties': {'p1': {'min': .5, 'max': 2., 'default': 1.}}
                                 if world in SCALAR_WORLDS else {},
            'probe_properties': {'p2' if world in SCALAR_WORLDS else 'mass':
                                 {'min': .5, 'max': 2. if world in SCALAR_WORLDS else 4., 'default': 1.}},
            'energy_budget': 100, 'max_probes_per_launch': 8,
            'max_readouts_per_probe': 50, 'observe_calls': 20,
            'max_times_per_observe': 50, 'wall_clock_seconds': 3600.,
            'first_light': first_light(world)}

def noise_sigma(world):
    """Same scale rule for every standard episode; clean validation is separate."""
    lo, hi, *_ = LIMITS[_world(world)]
    return float(.01*np.sqrt((lo*lo+hi*hi)/2))

def _cohort(world, radii, speeds, angle):
    angles = angle + np.arange(4)*(2*np.pi/4)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.column_stack((-np.sin(angles), np.cos(angles)))
    velocity = np.asarray(speeds)[:, None]*tangent + .025*np.sin(3*angles)[:, None]*unit
    attrs = [.7, 1., 1.3, 1.6] if world in SCALAR_WORLDS else [.5, 1., 2., 4.]
    return [_probe(world, p, v, a) for p, v, a in
            zip(np.asarray(radii)[:, None]*unit, velocity, attrs)]

def validation_cases(world):
    """Same schedule convention: 4 probes/case, 20 uniform positive offsets.

    A range describes legal initial placement, not a wall that confines a probe.
    No case speed is computed from the true force or an exact orbit condition.
    """
    world = _world(world)
    layouts = {
        '1': [([1.15, 1.4, 1.65, 1.9], [.4, .45, .5, .55]),
              ([2.6, 3.1, 3.6, 4.1], [.3, .4, .5, .45]),
              ([5.2, 5.9, 6.6, 7.3], [.3, .4, .5, .45])],
        '2': [([.8, 1.1, 1.4, 1.7], [.3, .4, .5, .6]),
              ([2.3, 2.6, 2.9, 3.2], [.4, .5, .6, .7]),
              ([4.4, 4.8, 5.2, 5.6], [.6, .7, .8, .9])],
        '3': [([.8, 1.1, 1.4, 1.7], [.3, .4, .5, .6]),
              ([2.3, 2.6, 2.9, 3.2], [.4, .5, .6, .7]),
              ([4.4, 4.8, 5.2, 5.6], [.6, .7, .8, .9])],
        '4': [([1.1, 1.35, 1.6, 1.85], [.4, .45, .5, .55]),
              ([2.5, 3., 3.5, 4.], [.35, .4, .45, .5]),
              ([5., 6., 7., 8.], [.25, .3, .35, .4])],
        '5': [([1., 1.3, 1.6, 1.9], [.4, .5, .6, .7]),
              ([2.4, 2.7, 3., 3.3], [.5, .6, .7, .8]),
              ([3.6, 3.9, 4.2, 4.5], [.6, .75, .9, 1.05])],
        '6': [([1.2, 1.6, 2., 2.4], [.35, .45, .55, .65]),
              ([3.6, 4.2, 4.8, 5.4], [.6, .7, .8, .9])],
    }
    duration = LIMITS[world][3]
    cases = []
    for band, (radii, speeds) in enumerate(layouts[world]):
        launches = (0., 4., 8., 12.) if world == '6' else (0., 4.)
        for k, launch in enumerate(launches):
            p1 = [.7, 1.7][k] if world in SCALAR_WORLDS else None
            case = {'name': f'b{band}-s{k}', 'group': f'band_{band}',
                    't_launch': launch,
                    'readout': (np.arange(1, 21)*(duration/20)).tolist(),
                    'source_properties': {'p1': p1} if p1 is not None else {},
                    'probes': _cohort(world, radii, speeds, .23+.57*k+.19*band)}
            cases.append(case)
    return copy.deepcopy(cases)
