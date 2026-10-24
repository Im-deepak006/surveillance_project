import sys
import os
import torch

# Ensure project root is importable, same effect as PYTHONPATH="."
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_on_video.run import MomentDETRPredictor
from extension.query_logic import fuse_candidate_ranks, parse_retrieval_query


def run_moment_detr(video_path, query_text, predictor):
    """
    Runs Moment-DETR inference and converts its output into the
    candidate format that rerank_with_attributes() expects.
    """
    predictions = predictor.localize_moment(
        video_path=video_path, query_list=[query_text]
    )
    # Confirmed format: list of [start, end, score]
    candidates = [
    {"start": max(0.0, start), "end": end, "temporal_score": score}
    for start, end, score in predictions[0]["pred_relevant_windows"]
    ]
    return candidates


def full_pipeline(video_path, query_text, ckpt_path):
    predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path="ViT-B/32",
        device="cpu" if not torch.cuda.is_available() else "cuda"
    )

    print(f"\nRunning Moment-DETR on query: \"{query_text}\"")
    candidates = run_moment_detr(video_path, query_text, predictor)
    for c in candidates:
        print(f"  Moment-DETR candidate: {c['start']:.1f}s - {c['end']:.1f}s, score={c['temporal_score']:.3f}")

    query_parts = parse_retrieval_query(query_text)
    # This module initializes YOLO/OpenCLIP when imported, so keep it lazy for
    # temporal-only or unsupported queries.
    if query_parts.appearance_text or query_parts.action_text or query_parts.carried_object_texts:
        from extension.attribute_matching import score_candidates_with_semantics

        scored = score_candidates_with_semantics(
            video_path,
            candidates,
            appearance_text=query_parts.appearance_text,
            action_text=query_parts.action_text,
            carried_object_texts=query_parts.carried_object_texts,
        )
    else:
        scored = [
            {
                **candidate,
                "appearance_score": None,
                "attribute_score": None,
                "appearance_evidence_status": "not_requested",
                "action_score": None,
                "action_evidence_status": "not_requested",
                "carried_object_score": None,
                "carried_object_detections": [],
                "carried_object_evidence_status": "not_requested",
            }
            for candidate in candidates
        ]
    final_results = fuse_candidate_ranks(scored, "temporal_appearance_action_carried_object")

    print("\n=== FINAL RESULTS ===")
    for r in final_results:
        print(f"[{r['start']:.1f}s - {r['end']:.1f}s]  "
              f"temporal={r['temporal_score']:.2f}  "
              f"appearance={r['appearance_score'] if r['appearance_score'] is not None else 'not scored'}  "
              f"action={r['action_score'] if r['action_score'] is not None else 'not scored'}  "
              f"objects={r['carried_object_score'] if r['carried_object_score'] is not None else 'not detected'}  "
              f"fused={r['fused_score'] if r['fused_score'] is not None else 'not scored'}")

    return final_results


if __name__ == "__main__":
    video_path = "run_on_video/example/RoripwjYFp8_60.0_210.0.mp4"
    query_text = "a woman in a black dress who cracks walnuts"
    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"
    full_pipeline(video_path, query_text, ckpt_path)
