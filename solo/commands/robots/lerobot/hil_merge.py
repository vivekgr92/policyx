"""
Merge a hardware-in-the-loop (HIL) correction session, recorded during
`solo robo --inference` with teleoperation override, into a base training
dataset - so corrections feed back into the next `solo robo --train` run
instead of sitting isolated in a local-only eval dataset.
"""

import typer


def merge_correction_into_dataset(base_repo_id: str, correction_repo_id: str, merged_repo_id: str) -> None:
    """
    Aggregate `base_repo_id` and `correction_repo_id` (a v3.0 LeRobot dataset each)
    into a new `merged_repo_id` dataset, then push the result to HuggingFace Hub.
    """
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    typer.echo(f"🔀 Merging '{correction_repo_id}' into '{base_repo_id}' -> '{merged_repo_id}'...")
    aggregate_datasets(repo_ids=[base_repo_id, correction_repo_id], aggr_repo_id=merged_repo_id)

    typer.echo(f"📤 Pushing merged dataset to HuggingFace Hub: {merged_repo_id}")
    merged_dataset = LeRobotDataset(merged_repo_id)
    merged_dataset.push_to_hub()
