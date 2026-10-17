"""Small hardware-free review-manifest and decision smoke check."""

from planner.contracts import Target
from planner.geometry import Region
from planner.review import (
    ManualTarget,
    ReviewContext,
    apply_review,
    create_manifest,
    make_decision,
)


def main() -> None:
    """Validate a saved review checkpoint and render its page."""
    context = ReviewContext("replay-1", "seeded", "stage-xy", "cal-1")
    candidate = Target("candidate-1", "seeded", "stage-xy", "cal-1", 10.0, 20.0)
    manifest = create_manifest(context, (candidate,))
    decision = make_decision(
        manifest,
        accepted_target_ids=(candidate.target_id,),
        additions=(ManualTarget("manual-1", 30.0, 40.0, "annotated spot"),),
    )
    result = apply_review(
        decision,
        (candidate,),
        context=context,
        region=Region(0, 0, 100, 100),
        max_targets=4,
    )
    print(
        f"manifest={result.manifest_sha256}; reviewed_targets={len(result.targets)}; manual={len(result.manual_provenance)}"
    )


if __name__ == "__main__":
    main()
