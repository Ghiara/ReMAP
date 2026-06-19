"""Compatibility wrapper for legacy imports.

Several baseline scripts used to import
`train.run_task_inference_high_level_policy_training`.
The active implementation now lives in
`train_multi_task_inference_high_level_policy.py`.
"""

from ReMAP.train.train_multi_task_inference_high_level_policy import (  # noqa: F401
    click_main,
    deep_update_dict,
    experiment,
    main,
    npify_dict,
)


if __name__ == "__main__":
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print(
        "Current device:",
        torch.cuda.current_device() if torch.cuda.is_available() else "CPU only",
    )
    print(
        "Device name:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
    )
    click_main()
