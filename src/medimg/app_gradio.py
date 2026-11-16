"""Lightweight Gradio web application for the Medical Imaging Triage Workstation."""
import os
import gradio as gr
from .infer import run_inference, load_models, DEFAULT_THRESHOLD

# Cache models in memory across requests
_CACHED_MODELS = None

def get_models():
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        _CACHED_MODELS = load_models()
    return _CACHED_MODELS

def predict_fn(image, threshold):
    if image is None:
        return None, None, "Please upload an image.", ""
    
    # Save uploaded PIL image to temporary path
    tmp_path = "tmp_gradio_input.png"
    image.save(tmp_path)
    
    models = get_models()
    result, overlay_img = run_inference(tmp_path, out_dir=None, models=models, threshold=threshold)
    
    # Extract mask
    mask_np = (overlay_img != image).any(axis=-1)
    
    # Format badge
    priority_color = "red" if result["priority"] == "HIGH PRIORITY" else "green"
    badge_html = f"""
    <div style="background-color: {'#ffebee' if result['priority'] == 'HIGH PRIORITY' else '#e8f5e9'};
                border: 2px solid {'#d32f2f' if result['priority'] == 'HIGH PRIORITY' else '#388e3c'};
                border-radius: 8px; padding: 16px; margin-bottom: 12px;">
        <h2 style="color: {'#c62828' if result['priority'] == 'HIGH PRIORITY' else '#2e7d32'}; margin: 0 0 8px 0;">
            {result['priority']}
        </h2>
        <p style="font-size: 18px; margin: 0;"><b>Melanoma Probability:</b> {result['risk_score']*100:.2f}%</p>
        <p style="font-size: 14px; color: #555; margin: 4px 0 0 0;">
            Decision threshold: {threshold:.3f} (target sensitivity &ge; 95% on validation)
        </p>
    </div>
    """
    
    # Latency table
    lat = result["latency_ms"]
    latency_str = f"Total: {lat['total']:.1f} ms | Preprocess: {lat['preprocess']:.1f} ms | Inference: {lat['inference']:.1f} ms | Postprocess: {lat['postprocess']:.1f} ms"
    
    return overlay_img, badge_html, result["summary"], latency_str

def build_app():
    # Find example images if available
    examples = []
    for d in ["data/isic_seg_det/images", "data/ham10000_raw"]:
        if os.path.exists(d):
            imgs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith((".png", ".jpg", ".jpeg"))][:3]
            for p in imgs:
                examples.append([p, DEFAULT_THRESHOLD])
            if examples:
                break

    with gr.Blocks(title="Medical Imaging Triage Workstation") as demo:
        gr.Markdown(
            """
            # Medical Imaging Triage and Lesion Analysis Workstation
            ### Dermatology Clinical Decision Support System
            **Disclaimer:** This workstation is an AI decision-support triage aid intended for clinician assistance.
            It does not provide autonomous medical diagnosis.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Dermoscopic Lesion Image")
                threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.01,
                    label="Triage Operating Threshold (Validation Target Sens >= 95%)"
                )
                submit_btn = gr.Button("Analyze Lesion & Triage", variant="primary")
                
                if examples:
                    gr.Examples(examples=examples, inputs=[input_image, threshold_slider])

            with gr.Column(scale=1):
                badge_output = gr.HTML(label="Triage Decision")
                overlay_output = gr.Image(type="pil", label="Multi-Task Overlay (Mask & Bounding Boxes)")
                latency_output = gr.Textbox(label="Inference Latency Breakdown", interactive=False)
                summary_output = gr.Textbox(label="Clinical Summary", lines=10, interactive=False)

        submit_btn.click(
            fn=predict_fn,
            inputs=[input_image, threshold_slider],
            outputs=[overlay_output, badge_output, summary_output, latency_output]
        )

    return demo

def launch(port=7860, share=False):
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=port, share=share)

if __name__ == "__main__":
    launch()
