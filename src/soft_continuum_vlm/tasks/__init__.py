"""Task definitions."""

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.feagine_metaworld_task import (
    FeagineMetaWorldTask,
    FeagineTaskEvaluation,
)
from soft_continuum_vlm.tasks.feagine_pick_place_tasks import (
    FEAGINE_PICK_PLACE_TASKS,
    PICK_PLACE_PHASES,
    FeaginePickLeftPlaceRightTask,
    FeaginePickPlaceTask,
    FeaginePickRightPlaceLeftTask,
    FeaginePickShelfPlaceShelfTask,
    make_feagine_pick_place_task,
)
from soft_continuum_vlm.tasks.feagine_push_tasks import (
    FEAGINE_PUSH_TASKS,
    FeagineContactPushTask,
    FeaginePushLeftToRightTask,
    FeaginePushRightToLeftTask,
    FeaginePushTask,
    make_feagine_push_task,
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


FEAGINE_METAWORLD_TASKS = {
    **FEAGINE_REACH_TASKS,
    **FEAGINE_PUSH_TASKS,
    **FEAGINE_PICK_PLACE_TASKS,
}


def make_feagine_metaworld_task(name: str) -> FeagineMetaWorldTask:
    try:
        task_type = FEAGINE_METAWORLD_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Feagine MetaWorld task: {name}") from exc
    return task_type()


__all__ = [
    "BaseTask",
    "ContactPushTask",
    "FEAGINE_METAWORLD_TASKS",
    "FEAGINE_PICK_PLACE_TASKS",
    "FEAGINE_PUSH_TASKS",
    "FEAGINE_REACH_TASKS",
    "FeagineContactPushTask",
    "FeagineMetaWorldTask",
    "FeaginePickLeftPlaceRightTask",
    "FeaginePickPlaceTask",
    "FeaginePickRightPlaceLeftTask",
    "FeaginePickShelfPlaceShelfTask",
    "FeaginePushLeftToRightTask",
    "FeaginePushRightToLeftTask",
    "FeaginePushTask",
    "FeagineReach3DTask",
    "FeagineReachLeftTask",
    "FeagineReachRightTask",
    "FeagineReachTask",
    "FeagineTaskEvaluation",
    "ObstacleAvoidPickTask",
    "PICK_PLACE_PHASES",
    "PickTask",
    "RotatePlaceTask",
    "TaskSpec",
    "make_feagine_metaworld_task",
    "make_feagine_pick_place_task",
    "make_feagine_push_task",
    "make_feagine_reach_task",
]
