"""Build a candidate initial-value case from permitted console metadata.

Observed/noisy trajectories are deliberately not used as initial states. This
helper knows neither a true acceleration law nor a hidden object's state.
"""
from __future__ import annotations
import copy


def case_from_observation(mission, experiment, *, source_target_id=None):
    """Convert a public launch record to the common candidate rollout input.

    source_target_id identifies the visible object whose *known control*
    properties were changed by this launch. A future console can supply it as
    mission['source_property_target_id']; it is never inferred from a law.
    """
    from .law import ValidationError
    from .prediction import validate_case
    if not isinstance(mission, dict) or not isinstance(experiment, dict):
        raise ValidationError('$', 'mission and experiment must be public dictionaries')
    try:
        initial=mission['background_initial_state']
        ids,positions,velocities=initial['ids'],initial['positions'],initial['velocities']
        properties=initial.get('properties',[{} for _ in ids])
        if not (len(ids)==len(positions)==len(velocities)==len(properties)):
            raise ValidationError('$.mission.background_initial_state', 'background arrays must have matching lengths')
        background=[{'id':body_id,'pos':copy.deepcopy(pos),'vel':copy.deepcopy(vel),
                     'properties':copy.deepcopy(props)}
                    for body_id,pos,vel,props in zip(ids,positions,velocities,properties)]
        controls=experiment.get('source_properties',{})
        if not isinstance(controls,dict):
            raise ValidationError('$.experiment.source_properties','must be a dictionary')
        if controls:
            target=source_target_id if source_target_id is not None else mission.get('source_property_target_id')
            if target not in ids:
                raise ValidationError('$.source_target_id','a known visible target ID is required for source controls')
            # A per-experiment intervention replaces the known default value;
            # this is official input construction, not a candidate override.
            background[ids.index(target)]['properties'].update(copy.deepcopy(controls))
        probes=[]
        for index,probe in enumerate(experiment['probes']):
            probes.append({'id':probe.get('id',f'P{index}'),'pos':copy.deepcopy(probe['pos']),
                           'vel':copy.deepcopy(probe['vel']),'properties':copy.deepcopy(probe['properties'])})
        case={'schema_version':'1.0.0','t_launch':experiment['t_launch'],
              'readout':copy.deepcopy(experiment['readout']),'background':background,'probes':probes}
    except (KeyError,TypeError,ValueError) as exc:
        if isinstance(exc,ValidationError):raise
        raise ValidationError('$','missing or malformed public mission/launch metadata') from exc
    return validate_case(case)


__all__=['case_from_observation']
