"""Task definitions."""

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.feagine_metaworld_task import (
    FeagineMetaWorldTask,
    FeagineTaskEvaluation,
)
from soft_continuum_vlm.tasks.feagine_reach_tasks import (
    FEAGINE_REACH_TASKS,
    FeagineReach3DTask,
    FeagineReachLeftTask,
    FeagineReachRightTask,
    FeagineReachTask,
    make_feagine_reach_task,
)
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask

__all__ = [
    "BaseTask",
    "ContactPushTask",
    "FEAGINE_REACH_TASKS",
    "FeagineMetaWorldTask",
    "FeagineReach3DTask",
    "FeagineReachLeftTask",
    "FeagineReachRightTask",
    "FeagineReachTask",
    "FeagineTaskEvaluation",
    "ObstacleAvoidPickTask",
    "PickTask",
    "RotatePlaceTask",
    "TaskSpec",
    "make_feagine_reach_task",
]
