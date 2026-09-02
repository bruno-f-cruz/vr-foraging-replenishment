"""Build the final stage of the ReplenishmentDepletionOffset curriculum with a custom reward amount.

Run with:
    uv run python make_stage.py
"""

import difflib
import sys
from pathlib import Path

from aind_behavior_curriculum import Stage, TrainerState
from aind_behavior_vr_foraging import task_logic as vr_task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic, AindVrForagingTaskParameters
from aind_behavior_vr_foraging_curricula.depletion import helpers
from aind_behavior_vr_foraging_curricula.depletion.metrics import metrics_from_dataset
from aind_behavior_vr_foraging_curricula.replenishment_depletion_offset import stages
from aind_behavior_vr_foraging_curricula.replenishment_depletion_offset.curriculum import CURRICULUM
from aind_behavior_vr_foraging_curricula.replenishment_depletion_offset.policies import p_update_replenishment_rate
from aind_behavior_vr_foraging_curricula.replenishment_depletion_offset.utils import make_patch

REWARD_AMOUNT_UL = 5.0  # force 5 uL, overriding the package's own default for this stage
OUTPUT_PATH = Path(__file__).parent / "mcm_final_stage.json"

PATCH_LABELS = ["High", "Medium", "Low"]


def make_final_stage(reward_amount: float = REWARD_AMOUNT_UL) -> Stage:
    """Rebuild `mcm_final_stage` from the package's own patch statistics, with a custom reward amount."""
    patches = [
        make_patch(
            label=label,
            state_index=i,
            odor_index=i,
            p_reward_max=stages.p_maxs[i],
            p_reward_min=stages.p_min[i],
            depletion_rate=stages.dep_rates[i],
            replenishment_rate=stages.rep_rates[i],
            n_states=stages.num_ps_states[i],
            rho=stages.rhos[i],
            replenishment_delay=stages.replenishment_delay[i],
            inter_patch_length=stages.interpatch_length[i],
            reward_amount=reward_amount,
        )
        for i, label in enumerate(PATCH_LABELS)
    ]

    environment_statistics = vr_task_logic.MarkovEnvironment(
        first_state_occupancy=[0.33, 0.33, 0.33],
        transition_matrix=[[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        patches=patches,
    )

    task_logic = AindVrForagingTaskLogic(
        task_parameters=AindVrForagingTaskParameters(
            rng_seed=None,
            environment=vr_task_logic.BlockStructure(
                blocks=[vr_task_logic.Block(environment=environment_statistics, end_conditions=[])],
                sampling_mode="Random",
            ),
            operation_control=helpers.make_default_operation_control(velocity_threshold=8),
        ),
        stage_name="mcm_final_stage",
    )

    return Stage(
        name="mcm_final_stage",
        task=task_logic,
        start_policies=[p_update_replenishment_rate],
        metrics_provider=metrics_from_dataset,
    )


def make_trainer_state(stage: Stage) -> TrainerState:
    """Wrap a `Stage` into the `TrainerState` envelope this script actually outputs.

    `curriculum` is the real, unmodified `ReplenishmentDepletionOffset` curriculum -- only
    `stage` carries the custom reward amount.
    """
    return TrainerState(
        curriculum=CURRICULUM,
        stage=stage,
        is_on_curriculum=True,
        active_policies=stage.start_policies,
    )


def diff_against_native(custom_json: str) -> str:
    """Unified diff between the package's native `mcm_final_stage` trainer state JSON and `custom_json`."""
    native_json = make_trainer_state(stages.make_s_mcm_final_stage()).model_dump_json(indent=2)
    diff = difflib.unified_diff(
        native_json.splitlines(keepends=True),
        custom_json.splitlines(keepends=True),
        fromfile="native (stages.make_s_mcm_final_stage)",
        tofile="custom (make_final_stage)",
    )
    return "".join(diff)


_RED = "\033[31m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def colorize_diff(diff: str) -> str:
    """Color a unified diff: green '+' lines, red '-' lines, cyan '@@' hunk headers."""
    colored_lines = []
    for line in diff.splitlines(keepends=True):
        if line.startswith(("+++", "---")):
            colored_lines.append(line)  # file headers stay plain
        elif line.startswith("@@"):
            colored_lines.append(f"{_CYAN}{line}{_RESET}")
        elif line.startswith("+"):
            colored_lines.append(f"{_GREEN}{line}{_RESET}")
        elif line.startswith("-"):
            colored_lines.append(f"{_RED}{line}{_RESET}")
        else:
            colored_lines.append(line)
    return "".join(colored_lines)


def main() -> None:
    stage = make_final_stage()
    custom_json = make_trainer_state(stage).model_dump_json(indent=2)
    OUTPUT_PATH.write_text(custom_json, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")

    diff = diff_against_native(custom_json)
    if diff:
        print("\nDiff vs native package stage (mcm_final_stage):")
        print(colorize_diff(diff) if sys.stdout.isatty() else diff)
    else:
        print("\nNo differences vs native package stage.")


if __name__ == "__main__":
    main()
