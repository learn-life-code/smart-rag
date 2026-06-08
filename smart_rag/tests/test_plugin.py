#!/usr/bin/env python3
"""Regression tests for the consumer-facing SmartRAG evidence plug-in."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_PASS = _FAIL = 0


def check(name, condition):
    global _PASS, _FAIL
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    _PASS += bool(condition)
    _FAIL += not condition


def main():
    from smart_rag import SmartRAG
    from smart_rag.core.fact import Fact
    from smart_rag.plugin import DistillRetriever, _baseline_db_path

    retriever = DistillRetriever.__new__(DistillRetriever)
    retriever.d = SmartRAG()
    retriever.stats = {}
    retriever.d.store.add_many([
        Fact(
            entity="VPE_K_VEHICLE_FRONT_VIEW_PRESENT",
            attribute="defined_in",
            value="service_vpe calibration publish on wake",
            source="vpe/include/vehicle_properties.h:42",
        ),
        Fact(
            entity="unrelated_camera_pipeline",
            attribute="event",
            value="camera frame received",
            source="camera.log:10",
        ),
    ])

    # Simulate a noisy broad semantic channel. Exact entity anchors must still
    # recover the cited baseline fact and the unrelated result must be rejected.
    retriever.search = lambda _query, top_k=8: [{
        "text": "unrelated camera frame received",
        "source": "camera.log:10",
        "score": 0.95,
        "kind": "prose",
    }][:top_k]

    evidence = retriever.expectation_evidence(
        "HUD calibration missing after wake",
        "Expected the HUD vehicle-front-view calibration to be published.",
        "VPE_K_VEHICLE_FRONT_VIEW_PRESENT service_vpe",
    )
    check("plugin: emits typed configuration evidence",
          "-- CONFIGURATION_AND_SUPPORT --" in evidence)
    check("plugin: emits typed execution-flow evidence",
          "-- EXECUTION_FLOW_AND_PRECONDITIONS --" in evidence)
    check("plugin: exact symbol survives noisy semantic retrieval",
          "VPE_K_VEHICLE_FRONT_VIEW_PRESENT" in evidence)
    check("plugin: every accepted fact remains source-cited",
          "[vpe/include/vehicle_properties.h:42]" in evidence)
    check("plugin: unrelated high-score camera result is rejected",
          "camera.log:10" not in evidence)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expected = root / ".distill" / "baseline_distill.db"
        check("plugin: build root resolves stable baseline location",
              _baseline_db_path(str(root)) == expected)

    print(f"\nPlugin: {_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
