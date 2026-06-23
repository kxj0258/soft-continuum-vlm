from __future__ import annotations

import numpy as np

from soft_continuum_vlm.data.dataset import DemoDataset
from soft_continuum_vlm.data.serialization import save_demo_npz


def test_demo_dataset_loads_npz_and_reports_dims(tmp_path) -> None:
    path = tmp_path / "demo.npz"
    save_demo_npz(
        path,
        proprioception=np.zeros((2, 8), dtype=np.float32),
        contact=np.zeros((2, 3), dtype=np.float32),
        language=np.array(["pick", "pick"]),
        language_feature=np.zeros((2, 64), dtype=np.float32),
        morphology=np.zeros((2, 4), dtype=np.float32),
        action=np.array(["{}", "{}"]),
        action_vector=np.zeros((2, 8), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        done=np.array([False, True]),
        success=np.array([False, True]),
        task_name=np.array(["pick_red_object", "pick_red_object"]),
        phase=np.array(["scripted", "scripted"]),
        episode_id=np.array([0, 0], dtype=np.int32),
        step_id=np.array([0, 1], dtype=np.int32),
    )

    dataset = DemoDataset(path)

    assert len(dataset) == 2
    assert dataset.action_dim() == 8
    assert dataset.proprioception_dim() == 8
    assert dataset.contact_dim() == 3
    assert dataset.morphology_dim() == 4
    assert dataset[1]["done"] is True
