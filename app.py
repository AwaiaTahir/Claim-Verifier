"""Gradio entry point and end-to-end pipeline for claim verification."""

from __future__ import annotations

import html
import time
from typing import Any

import gradio as gr
import requests
import torch

from config import (
    CLAIM_AMBIGUOUS,
    CLAIM_COMPOUND,
    CLAIM_OPINION,
    CONFLICTING_EVIDENCE,
    EXAMPLE_CLAIMS,
    NOT_ENOUGH_EVIDENCE,
    NOT_VERIFIABLE,
    OLLAMA_MODEL,
    OLLAMA_OK_PROMPT,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    CLAIM_FACTUAL,
    REFUTED,
    STANCE_COLORS,
    STANCE_NEUTRAL,
    SUPPORTED,
    TOP_K_RERANK,
    VERDICT_COLORS,
)
from modules.claim_classifier import classify_claim
from modules.explainer import format_output, generate_explanation
from modules.query_rewriter import rewrite_to_queries
from modules.reranker import get_nli_model, get_reranker_model, label_stance, rerank
from modules.llm_client import set_llm_available
from modules.retriever import IterativeRetriever, get_embedding_model
from modules.verdict import passes_retrieval_confidence, predict_verdict

iterative_retriever = IterativeRetriever()


def startup_gpu_check() -> None:
    """Print the GPU status before launching the Gradio application."""
    print(f"[GPU] PyTorch version: {torch.__version__}")
    print(f"[GPU] PyTorch CUDA build: {torch.version.cuda or 'CPU-only torch build'}")
    if torch.cuda.is_available():
        print(f"[GPU] Using: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[WARNING] CUDA not available to PyTorch - running on CPU, will be slow")
        if torch.version.cuda is None:
            print("[GPU DIAGNOSTIC] Install the CUDA PyTorch wheel inside this .venv.")
        else:
            print("[GPU DIAGNOSTIC] PyTorch has CUDA support, but the NVIDIA driver/GPU was not visible.")


def test_ollama() -> None:
    """Check whether the configured Ollama model is reachable before launch."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": OLLAMA_OK_PROMPT, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            set_llm_available(True)
            print(f"[Ollama] {OLLAMA_MODEL} is running and responding correctly")
        else:
            set_llm_available(False)
            print(f"[Ollama WARNING] Status {response.status_code} - LLM features will use fallbacks")
    except Exception as exc:
        set_llm_available(False)
        print(f"[Ollama WARNING] Not reachable: {exc} - LLM features will use fallbacks")


def warmup_all_models() -> None:
    """Load transformer models before Gradio launch while keeping imports lazy."""
    steps = [
        ("embedding model", get_embedding_model),
        ("reranker model", get_reranker_model),
        ("NLI model", get_nli_model),
    ]
    for index, (name, loader) in enumerate(steps, start=1):
        print(f"[startup] Loading {name} ({index}/{len(steps)})")
        loader()


def _empty_result(claim: str, start_time: float) -> dict[str, Any]:
    """Return a structured result for empty input."""
    return {
        "claim": claim,
        "verdict": NOT_ENOUGH_EVIDENCE,
        "confidence": 0.0,
        "explanation": "Enter a factual claim so the system can retrieve evidence.",
        "evidence": [],
        "queries_used": [],
        "claim_type": "",
        "processing_time_seconds": round(time.time() - start_time, 2),
        "reason": "No claim was provided.",
    }


def _ensure_neutral_stance(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach neutral stance labels to passages that skipped NLI due to low confidence."""
    labeled = []
    for passage in passages:
        item = dict(passage)
        item.setdefault("stance", STANCE_NEUTRAL)
        item.setdefault(
            "stance_scores",
            {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
        )
        labeled.append(item)
    return labeled


def verify_single_claim(
    claim: str,
    start_time: float | None = None,
) -> dict[str, Any]:
    """Verify one atomic claim and return a structured verdict result."""
    if start_time is None:
        start_time = time.time()

    print(f"[Pipeline] Verifying single claim: {claim}")
    queries = rewrite_to_queries(claim)
    raw_passages = iterative_retriever.retrieve(claim, queries)
    retrieval_queries = getattr(iterative_retriever, "last_queries_used", queries) or queries
    print(f"[Pipeline] Retrieved {len(raw_passages)} raw passages")
    reranked = rerank(claim, raw_passages, top_k=TOP_K_RERANK)
    print(f"[Pipeline] Reranked to {len(reranked)} passages")

    if passes_retrieval_confidence(reranked):
        reranked_with_stance = label_stance(claim, reranked)
        print("[Pipeline] NLI stance labeling completed")
    else:
        reranked_with_stance = _ensure_neutral_stance(reranked)
        print("[Pipeline] NLI skipped because retrieval confidence was low")

    verdict_result = predict_verdict(claim, reranked_with_stance)
    explanation = generate_explanation(claim, verdict_result, reranked_with_stance[:3])

    verdict_result["processing_time_seconds"] = round(time.time() - start_time, 2)
    verdict_result["queries_used"] = retrieval_queries
    verdict_result["claim_type"] = verdict_result.get("claim_type", CLAIM_FACTUAL)

    return format_output(claim, verdict_result, reranked_with_stance, explanation)


def _aggregate_compound_verdict(results: list[dict[str, Any]]) -> str:
    """Choose an overall verdict for several verified sub-claims."""
    verdicts = [result.get("verdict", NOT_ENOUGH_EVIDENCE) for result in results]
    if not verdicts:
        return NOT_ENOUGH_EVIDENCE
    if any(verdict == CONFLICTING_EVIDENCE for verdict in verdicts):
        return CONFLICTING_EVIDENCE
    if SUPPORTED in verdicts and any(verdict != SUPPORTED for verdict in verdicts):
        return CONFLICTING_EVIDENCE
    if all(verdict == SUPPORTED for verdict in verdicts):
        return SUPPORTED
    if any(verdict == REFUTED for verdict in verdicts):
        return REFUTED
    return NOT_ENOUGH_EVIDENCE


def format_compound_result(claim: str, results: list[dict[str, Any]], start_time: float) -> dict[str, Any]:
    """Combine multiple sub-claim results into one UI-friendly response."""
    overall = _aggregate_compound_verdict(results)
    confidence_values = [float(result.get("confidence", 0.0)) for result in results]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    evidence = []
    queries = []
    explanation_parts = []
    for index, result in enumerate(results, start=1):
        explanation_parts.append(f"Sub-claim {index}: {result.get('verdict')} - {result.get('explanation')}")
        queries.extend(result.get("queries_used", []))
        for item in result.get("evidence", []):
            if len(evidence) < 5:
                evidence.append(dict(item))
    for rank, item in enumerate(evidence, start=1):
        item["rank"] = rank
    return {
        "claim": claim,
        "verdict": overall,
        "confidence": round(confidence, 2),
        "explanation": " ".join(explanation_parts),
        "evidence": evidence,
        "queries_used": queries,
        "claim_type": CLAIM_COMPOUND,
        "processing_time_seconds": round(time.time() - start_time, 2),
        "reason": "Compound claim verified by checking each sub-claim separately.",
    }
def verify_single_claim_stream(
    claim: str,
    start_time: float | None = None,
):
    """Generator that yields intermediate verification states for a single claim."""
    if start_time is None:
        start_time = time.time()

    yield {"step": "rewriting_queries", "message": "Rewriting claim into search queries...", "progress": 0.1}
    queries = rewrite_to_queries(claim)
    
    yield {"step": "retrieving", "message": "Searching local database and web...", "progress": 0.3, "queries": queries}
    raw_passages = iterative_retriever.retrieve(claim, queries)
    retrieval_queries = getattr(iterative_retriever, "last_queries_used", queries) or queries
    
    yield {"step": "reranking", "message": f"Reranking {len(raw_passages)} passages...", "progress": 0.6, "queries": retrieval_queries}
    reranked = rerank(claim, raw_passages, top_k=TOP_K_RERANK)
    
    yield {"step": "nli", "message": "Running NLI stance labeling...", "progress": 0.75, "queries": retrieval_queries, "evidence": reranked}
    if passes_retrieval_confidence(reranked):
        reranked_with_stance = label_stance(claim, reranked)
    else:
        reranked_with_stance = _ensure_neutral_stance(reranked)
        
    yield {"step": "verdict", "message": "Predicting final verdict...", "progress": 0.85, "queries": retrieval_queries, "evidence": reranked_with_stance}
    verdict_result = predict_verdict(claim, reranked_with_stance)
    
    yield {"step": "explaining", "message": "Generating explanation...", "progress": 0.95, "queries": retrieval_queries, "evidence": reranked_with_stance, "verdict": verdict_result}
    explanation = generate_explanation(claim, verdict_result, reranked_with_stance[:3])
    
    verdict_result["processing_time_seconds"] = round(time.time() - start_time, 2)
    verdict_result["queries_used"] = retrieval_queries
    verdict_result["claim_type"] = verdict_result.get("claim_type", CLAIM_FACTUAL)
    
    final_result = format_output(claim, verdict_result, reranked_with_stance, explanation)
    yield {"step": "done", "message": "Done", "progress": 1.0, "result": final_result}


def verify_claim_stream(claim: str):
    """Generator that yields intermediate states for the full claim verification pipeline."""
    start_time = time.time()
    cleaned_claim = (claim or "").strip()
    if not cleaned_claim:
        yield {"step": "done", "progress": 1.0, "result": _empty_result(cleaned_claim, start_time)}
        return

    yield {"step": "classifying", "message": "Classifying claim type...", "progress": 0.05}
    classification = classify_claim(cleaned_claim)
    claim_type = classification.get("type", "")
    is_opinion = claim_type == CLAIM_OPINION

    if not classification.get("is_checkable", True) or is_opinion:
        msg = classification.get("message") or (
            "This claim is a subjective opinion and cannot be objectively verified."
            if is_opinion
            else "This claim cannot be verified as stated."
        )
        verdict = NOT_VERIFIABLE if is_opinion else NOT_ENOUGH_EVIDENCE
        res = {
            "claim": cleaned_claim,
            "verdict": verdict,
            "confidence": 0.0,
            "explanation": msg,
            "evidence": [],
            "queries_used": [],
            "claim_type": claim_type,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "reason": msg,
        }
        yield {"step": "done", "progress": 1.0, "result": res}
        return

    if classification.get("type") == CLAIM_COMPOUND and classification.get("sub_claims"):
        sub_claims = classification["sub_claims"]
        results = []
        for i, sub_claim in enumerate(sub_claims):
            yield {"step": "sub_claim", "message": f"Verifying sub-claim {i+1} of {len(sub_claims)}...", "progress": 0.1 + (0.8 * i / len(sub_claims))}
            results.append(verify_single_claim(sub_claim))
        res = format_compound_result(cleaned_claim, results, start_time)
        yield {"step": "done", "progress": 1.0, "result": res}
        return

    for state in verify_single_claim_stream(cleaned_claim, start_time):
        state["claim_type"] = claim_type
        yield state

def verify_claim(claim: str) -> dict[str, Any]:
    """Run the full claim verification pipeline for a free-text user claim."""
    start_time = time.time()
    cleaned_claim = (claim or "").strip()
    print("=" * 72)
    print(f"[Pipeline] New claim: {cleaned_claim}")
    if not cleaned_claim:
        return _empty_result(cleaned_claim, start_time)

    classification = classify_claim(cleaned_claim)
    claim_type = classification.get("type", "")
    # Only block opinion claims — ambiguous claims still go through the pipeline
    # (factual myths get mis-tagged as 'ambiguous' by Ollama but are properly REFUTED)
    is_opinion = claim_type == CLAIM_OPINION

    if not classification.get("is_checkable", True) or is_opinion:
        msg = classification.get("message") or (
            "This claim is a subjective opinion and cannot be objectively verified."
            if is_opinion
            else "This claim cannot be verified as stated."
        )
        verdict = NOT_VERIFIABLE if is_opinion else NOT_ENOUGH_EVIDENCE
        print(f"[Classifier] type={claim_type} — returning {verdict}")
        return {
            "claim": cleaned_claim,
            "verdict": verdict,
            "confidence": 0.0,
            "explanation": msg,
            "evidence": [],
            "queries_used": [],
            "claim_type": claim_type,
            "processing_time_seconds": round(time.time() - start_time, 2),
            "reason": msg,
        }


    if classification.get("type") == CLAIM_COMPOUND and classification.get("sub_claims"):
        results = [
            verify_single_claim(sub_claim)
            for sub_claim in classification["sub_claims"]
        ]
        return format_compound_result(cleaned_claim, results, start_time)

    result = verify_single_claim(cleaned_claim, start_time)
    result["claim_type"] = classification.get("type", result.get("claim_type", ""))
    print(
        "[Pipeline] Done "
        f"verdict={result['verdict']} confidence={result['confidence']:.2f} "
        f"evidence={len(result['evidence'])} time={result['processing_time_seconds']}s"
    )
    return result


def render_verdict_html(output: dict[str, Any]) -> str:
    """Render the verdict as a large colored HTML label."""
    verdict = str(output.get("verdict", NOT_ENOUGH_EVIDENCE))
    color = VERDICT_COLORS.get(verdict, "#667085")
    safe_verdict = html.escape(verdict)
    return (
        f"<div class='verdict-card' style='border-color:{color};'>"
        f"<div class='verdict-label' style='color:{color};'>{safe_verdict}</div>"
        "</div>"
    )


def render_claim_type_html(output: dict[str, Any]) -> str:
    """Render the claim classification badge."""
    claim_type = html.escape(str(output.get("claim_type", "") or "unknown"))
    return f"<span class='claim-type-badge'>{claim_type}</span>"


def render_processing_time_html(output: dict[str, Any]) -> str:
    """Render the processing time field."""
    seconds = float(output.get("processing_time_seconds", 0.0))
    return f"<div class='metric-line'>Processing time: <strong>{seconds:.2f}s</strong></div>"


def _truncate_for_card(text: str, limit: int = 300) -> tuple[str, bool]:
    """Return a truncated card preview and whether it needs expansion."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned, False
    return cleaned[:limit].rstrip() + "...", True


def _score_bar_color(score: float) -> str:
    """Return a color for the score bar based on the score value."""
    if score > 0.7:
        return "#16803c"
    elif score > 0.4:
        return "#b54708"
    return "#b42318"


def _credibility_badge_html(item: dict[str, Any]) -> str:
    """Render a credibility badge for a passage (BONUS FIX)."""
    label = item.get("credibility_label", "standard")
    weight = float(item.get("credibility_weight", 1.0))
    if label == "trusted":
        return f"<span class='credibility-badge cred-trusted'>⭐ Trusted ({weight:.1f}x)</span>"
    elif label == "low":
        return f"<span class='credibility-badge cred-low'>⚠ Low ({weight:.1f}x)</span>"
    return ""


def render_evidence_html(output: dict[str, Any]) -> str:
    """Render the top evidence passages as HTML cards."""
    evidence = output.get("evidence", [])
    if not evidence:
        return "<div class='empty-evidence'>No evidence passages available.</div>"

    cards = []
    for item in evidence:
        stance = str(item.get("stance", STANCE_NEUTRAL) or STANCE_NEUTRAL)
        stance_color = STANCE_COLORS.get(stance, "#667085")
        score = float(item.get("relevance_score", 0.0))
        score_pct = round(score * 100, 1)
        bar_color = _score_bar_color(score)
        source = html.escape(str(item.get("source", "") or "source"))
        url = str(item.get("url", "") or "")
        text = str(item.get("text", "") or "")
        preview, expanded = _truncate_for_card(text)
        safe_preview = html.escape(preview)
        rank = int(item.get("rank", 0))
        cred_badge = _credibility_badge_html(item)
        if url:
            source_html = f"<a href='{html.escape(url)}' target='_blank'>{source}</a>{cred_badge}"
        else:
            source_html = f"{source}{cred_badge}"
        if expanded:
            body = (
                f"<p>{safe_preview}</p>"
                f"<details><summary>Expand</summary><p>{html.escape(text)}</p></details>"
            )
        else:
            body = f"<p>{safe_preview}</p>"
        # --- FIX 7+8: Score as percentage with colored bar ---
        score_display = (
            f"<div class='score-container'>"
            f"<span class='score-pill'>{score_pct}%</span>"
            f"<div class='score-bar'><div class='score-fill' style='width:{score_pct}%; background:{bar_color};'></div></div>"
            f"</div>"
        )
        cards.append(
            "<article class='evidence-card'>"
            "<div class='evidence-topline'>"
            f"<span class='rank-pill'>#{rank}</span>"
            f"<span class='stance-pill' style='background:{stance_color};'>{html.escape(stance)}</span>"
            f"{score_display}"
            "</div>"
            f"<div class='source-line'>{source_html}</div>"
            f"{body}"
            "</article>"
        )
    return "".join(cards)


def render_queries_html(output: dict[str, Any]) -> str:
    """Render the rewritten queries used by retrieval."""
    queries = output.get("queries_used", [])
    if not queries:
        return "<div class='query-list'>No queries were used.</div>"
    items = "".join(f"<li>{html.escape(str(query))}</li>" for query in queries)
    return f"<ol class='query-list'>{items}</ol>"


def verify_claim_ui(claim: str):
    """Run verification for the Gradio UI and yield component values iteratively."""
    for state in verify_claim_stream(claim):
        pct = state.get("progress", 0.0)
        msg = state.get("message", "Processing...")
        
        if state["step"] == "done":
            output = state["result"]
            confidence_percent = round(float(output.get("confidence", 0.0)) * 100, 2)
            yield (
                render_verdict_html(output),
                confidence_percent,
                render_claim_type_html(output),
                render_processing_time_html(output),
                str(output.get("explanation", "")),
                render_evidence_html(output),
                render_queries_html(output),
            )
        else:
            progress_html = f"""
            <div style="padding: 20px; border: 2px solid #344054; border-radius: 8px; background: #1d2939; color: white;">
               <h3 style="margin-top: 0; font-size: 18px;">Verifying Claim...</h3>
               <div style="width: 100%; background-color: #101828; border-radius: 4px; margin: 15px 0;">
                  <div style="height: 8px; background-color: #f97316; border-radius: 4px; width: {pct * 100}%; transition: width 0.5s ease-in-out;"></div>
               </div>
               <p style="margin-bottom: 0; color: #d0d5dd; font-size: 14px;">⏳ {msg}</p>
            </div>
            """
            
            temp_output = {
                "evidence": state.get("evidence", []),
                "queries_used": state.get("queries", []),
            }
            yield (
                progress_html,
                0,
                "",
                "",
                "Analyzing...",
                render_evidence_html(temp_output) if state.get("evidence") else "",
                render_queries_html(temp_output) if state.get("queries") else "",
            )


def make_example_handler(example: str):
    """Create a small handler that fills the claim textbox with an example."""

    def fill_example() -> str:
        """Return the stored example claim."""
        return example

    return fill_example


def build_ui() -> gr.Blocks:
    """Build the required Gradio Blocks two-column interface."""
    css = """
    .verdict-card { border: 2px solid; border-radius: 8px; padding: 18px; background: #ffffff; color: #101828; }
    .verdict-label { font-size: 28px; font-weight: 800; letter-spacing: 0; }
    .claim-type-badge { display: inline-block; padding: 6px 10px; border: 1px solid #d0d5dd; border-radius: 8px; background: #f9fafb; color: #344054; font-weight: 700; }
    .metric-line { color: #d0d5dd; font-size: 14px; }
    .metric-line strong { color: #ffffff; }
    .evidence-card { border: 1px solid #344054; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #1d2939; color: #f9fafb; }
    .evidence-card p { color: #f9fafb; opacity: 1; }
    .evidence-card details, .evidence-card summary { color: #d0d5dd; }
    .evidence-topline { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
    .rank-pill { border: 1px solid #667085; border-radius: 999px; padding: 3px 8px; font-size: 12px; color: #f9fafb; background: #344054; }
    .score-pill { border: 1px solid #667085; border-radius: 999px; padding: 3px 8px; font-size: 12px; color: #f9fafb; background: #344054; min-width: 80px; }
    .stance-pill { border-radius: 999px; padding: 4px 8px; color: #ffffff; font-size: 12px; font-weight: 800; }
    .source-line { font-size: 13px; margin-bottom: 8px; color: #d0d5dd; }
    .source-line a { color: #93c5fd; }
    .empty-evidence { border: 1px dashed #667085; border-radius: 8px; padding: 14px; color: #d0d5dd; }
    .query-list { margin: 0; padding-left: 20px; color: #f9fafb; }
    .score-bar { height: 8px; background: #344054; border-radius: 4px; overflow: hidden; margin-top: 2px; width: 100%; }
    .score-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    .score-container { display: flex; align-items: center; gap: 6px; min-width: 120px; }
    .credibility-badge { border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 700; display: inline-block; margin-left: 4px; }
    .cred-trusted { background: #16803c; color: #ffffff; }
    .cred-standard { background: #667085; color: #f9fafb; }
    .cred-low { background: #b42318; color: #ffffff; }
    """

    with gr.Blocks(css=css, title="Claim Verification System") as demo:
        gr.Markdown("# Claim Verification System")
        with gr.Row():
            with gr.Column(scale=1):
                import config
                backend_select = gr.Radio(
                    choices=["groq", "ollama"],
                    value=config.LLM_BACKEND,
                    label="LLM Backend",
                    info="Choose between local Ollama or cloud Groq API",
                )
                
                def _update_backend(val):
                    config.LLM_BACKEND = val
                
                backend_select.change(fn=_update_backend, inputs=[backend_select], outputs=[])

                claim_input = gr.Textbox(
                    label="Enter a claim to verify",
                    placeholder="e.g. The Great Wall of China is visible from space",
                    lines=4,
                )
                verify_button = gr.Button("Verify Claim", variant="primary")
                verdict_display = gr.HTML()
                confidence_bar = gr.Slider(
                    label="Confidence",
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False,
                )
                claim_type_badge = gr.HTML()
                processing_time_display = gr.HTML()
                explanation_box = gr.Textbox(
                    label="Explanation",
                    interactive=False,
                    lines=5,
                    max_lines=10,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Retrieved Evidence")
                evidence_display = gr.HTML()
                with gr.Accordion("Queries used", open=False):
                    queries_display = gr.HTML()

        with gr.Row():
            for example in EXAMPLE_CLAIMS:
                gr.Button(example).click(
                    fn=make_example_handler(example),
                    inputs=[],
                    outputs=[claim_input],
                    show_progress=False,
                )

        verify_button.click(
            fn=verify_claim_ui,
            inputs=[claim_input],
            outputs=[
                verdict_display,
                confidence_bar,
                claim_type_badge,
                processing_time_display,
                explanation_box,
                evidence_display,
                queries_display,
            ],
            show_progress=False,
        )
    return demo


def main() -> None:
    """Run startup checks, warm models, and launch the Gradio server."""
    startup_gpu_check()
    test_ollama()
    warmup_all_models()
    demo = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
