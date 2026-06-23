"""Task definitions."""

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask

__all__ = [
    "BaseTask",
    "ContactPushTask",
    "ObstacleAvoidPickTask",
    "PickTask",
    "RotatePlaceTask",
    "TaskSpec",
]
