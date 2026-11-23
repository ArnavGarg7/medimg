"""Interactive Multi-Tab Web Application for Medical Imaging Triage & Lesion Analysis Workstation.
Run with:
    python -m medimg gui --port 7860
"""
import os, json
import gradio as gr
from PIL import Image
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
        return None, "<div style='color:#f59e0b; padding:10px;'>Please upload or select an image.</div>", "Please upload an image.", ""
    
    try:
        # Save uploaded PIL image to temporary path
        tmp_path = "tmp_gradio_input.png"
        image.save(tmp_path)
        
        models = get_models()
        result, overlay_img = run_inference(tmp_path, out_dir=None, models=models, threshold=threshold)
        
        # Format triage badge HTML
        is_high = result["priority"] == "HIGH PRIORITY"
        bg_color = "#ffebee" if is_high else "#e8f5e9"
        border_color = "#d32f2f" if is_high else "#388e3c"
        text_color = "#c62828" if is_high else "#2e7d32"
        badge_label = "HIGH PRIORITY (Fast-Track Urgent Dermatologist Review)" if is_high else "LOW PRIORITY (Routine Clinical Follow-Up)"

        badge_html = f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color};
                    border-radius: 8px; padding: 16px; margin-bottom: 12px; font-family: sans-serif;">
            <h2 style="color: {text_color}; margin: 0 0 8px 0; font-size: 22px;">
                {badge_label}
            </h2>
            <div style="font-size: 18px; margin: 6px 0;">
                <b>Melanoma Risk Probability:</b> <span style="font-size: 22px; font-weight: bold; color: {text_color};">{result['risk_score']*100:.2f}%</span>
            </div>
            <div style="font-size: 14px; color: #555; margin-top: 4px;">
                <b>Operating Point:</b> Threshold = {threshold:.3f} (Calibrated on validation cohort for &ge; 95% target sensitivity)
            </div>
            <div style="font-size: 14px; color: #555; margin-top: 2px;">
                <b>Lesion Footprint:</b> {result['lesion_area_pct']:.2f}% of field of view | <b>Suspicious Foci:</b> {result['detection_count']} box(es) detected
            </div>
        </div>
        """
        
        # Latency table
        lat = result["latency_ms"]
        latency_str = f"Total: {lat['total']:.1f} ms  |  Preprocess: {lat['preprocess']:.1f} ms  |  Inference: {lat['inference']:.1f} ms  |  Postprocess: {lat['postprocess']:.1f} ms"
        
        return overlay_img, badge_html, result["summary"], latency_str
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        badge_err = f"<div style='background:#fee2e2; border:2px solid #ef4444; border-radius:8px; padding:12px; color:#b91c1c;'><b>Inference Error:</b> {str(e)}</div>"
        return None, badge_err, f"Error during inference:\n{err_detail}", "Error"

def build_app():
    # Find example images (prioritize real clinical samples)
    examples = []
    for d in ["samples", "data/ham10000/images", "data/isic_seg_det/images", "data/synth/images"]:
        if os.path.exists(d):
            imgs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith((".png", ".jpg", ".jpeg"))][:4]
            for p in imgs:
                examples.append([p, DEFAULT_THRESHOLD])
            if examples:
                break

    with gr.Blocks(title="Medical Imaging Triage Workstation") as demo:
        gr.Markdown(
            """
            # Medical Imaging Triage and Lesion Analysis Workstation
            ### Dermatology Clinical Decision Support System (SaMD Prototype)
            **Clinical Notice:** This system is an automated triage decision-support aid designed for specialist assistance.
            It does **not** replace a clinician and does **not** provide autonomous medical diagnosis.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Interactive Workstation
            with gr.Tab("Interactive Triage Workstation"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_image = gr.Image(type="pil", label="Upload Dermoscopic Lesion Image")
                        threshold_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.01,
                            label="Triage Operating Threshold (Val Target Sensitivity >= 95%)",
                            info="Scores at or above this threshold trigger HIGH PRIORITY triage."
                        )
                        submit_btn = gr.Button("Analyze Lesion & Generate Triage Report", variant="primary", size="lg")
                        
                        if examples:
                            gr.Examples(
                                examples=examples,
                                inputs=[input_image, threshold_slider],
                                label="Sample Real Clinical Lesions (ISIC / HAM10000)"
                            )

                    with gr.Column(scale=6):
                        badge_output = gr.HTML(label="Triage Decision")
                        overlay_output = gr.Image(type="pil", label="Multi-Task Visual Overlay (Lesion Contour & Bounding Boxes)")
                        latency_output = gr.Textbox(label="Execution Latency Breakdown", interactive=False)
                        summary_output = gr.Textbox(label="Clinical Decision Support Summary", lines=8, interactive=False)

                submit_btn.click(
                    fn=predict_fn,
                    inputs=[input_image, threshold_slider],
                    outputs=[overlay_output, badge_output, summary_output, latency_output]
                )

            # Tab 2: Clinical Evaluation & Performance Scorecard
            with gr.Tab("Model Performance & Evaluation Scorecard"):
                gr.Markdown(
                    """
                    ### Held-Out Test Set Evaluation ($N=390$ Classification, $N=90$ Segmentation & Detection)
                    Models evaluated strictly once on held-out test data with threshold locked on validation data.
                    """
                )
                with gr.Row():
                    gr.HTML(
                        """
                        <table style="width:100%; border-collapse: collapse; text-align: left; font-family: sans-serif;">
                            <tr style="background-color: #f5f5f5; border-bottom: 2px solid #ddd;">
                                <th style="padding: 10px;">Pipeline Task</th>
                                <th style="padding: 10px;">Primary Architecture</th>
                                <th style="padding: 10px;">Test Performance Metric</th>
                                <th style="padding: 10px;">CPU Latency (Median)</th>
                                <th style="padding: 10px;">Acceptance Status</th>
                            </tr>
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td style="padding: 10px;"><b>Triage Classification</b></td>
                                <td style="padding: 10px;">ResNet-18 (Pretrained)</td>
                                <td style="padding: 10px;">AUROC: <b>0.8984</b> (95% CI: [0.869, 0.925])<br>Sens: <b>94.61%</b>, Spec: <b>68.61%</b></td>
                                <td style="padding: 10px;"><b>17.35 ms</b> (PyTorch)<br><b>6.76 ms</b> (ONNX 2.55x)</td>
                                <td style="padding: 10px; color: green;"><b>PASSED</b> (Target &ge; 0.85)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td style="padding: 10px;"><b>Lesion Boundary Segmentation</b></td>
                                <td style="padding: 10px;">ResNetUNet (Pretrained)</td>
                                <td style="padding: 10px;">Test Dice: <b>0.8989</b><br>Test IoU: <b>0.8282</b></td>
                                <td style="padding: 10px;"><b>32.68 ms</b></td>
                                <td style="padding: 10px; color: green;"><b>PASSED</b> (Target &ge; 0.80)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td style="padding: 10px;"><b>Lesion Focus Detection</b></td>
                                <td style="padding: 10px;">RetinaNet (ResNet50-FPN)</td>
                                <td style="padding: 10px;">Test AP50: <b>0.9794</b><br>Test AP50:95: <b>0.5331</b></td>
                                <td style="padding: 10px;"><b>99.30 ms</b></td>
                                <td style="padding: 10px; color: green;"><b>PASSED</b> (Target &ge; 0.50)</td>
                            </tr>
                        </table>
                        """
                    )
                with gr.Row():
                    if os.path.exists("out/evaluation/calibration_curve.png"):
                        gr.Image("out/evaluation/calibration_curve.png", label="Reliability Diagram (Calibration Curve, ECE = 0.140)")
                    if os.path.exists("out/evaluation/confusion_matrix.png"):
                        gr.Image("out/evaluation/confusion_matrix.png", label="Test Confusion Matrix (TP: 158, FN: 9, FP: 70, TN: 153)")

            # Tab 3: Robustness & Clinical Stress Testing
            with gr.Tab("Robustness & Artifact Stress Tests"):
                gr.Markdown(
                    """
                    ### Clinical Stress Testing: Metric Degradation vs Image Artifact Severity
                    Evaluated against 6 synthetic clinical imaging artifacts on the real test split.
                    """
                )
                if os.path.exists("out/robustness/robustness_curves.png"):
                    gr.Image("out/robustness/robustness_curves.png", label="Degradation Curves Across 6 Clinical Perturbations")
                gr.Markdown(
                    """
                    **Key Clinical Robustness Takeaways:**
                    - **JPEG Quality & Resolution:** Highly robust down to Quality 10 and 0.25x downsampling (AUROC remains $> 0.88$). Safe for tele-dermatology compression.
                    - **Illumination Dimming:** Severe collapse below 0.6x intensity factor (AUROC drops to 0.52). Dermatoscope illumination calibration is critical.
                    - **Blur:** Sensitivity drops from 94.6% to 64.7% at 4px blur radius due to edge loss. Out-of-focus images must be re-acquired.
                    """
                )

            # Tab 4: Grad-CAM Interpretability & Subgroup Parity
            with gr.Tab("Interpretability & Subgroup Demographic Parity"):
                gr.Markdown(
                    """
                    ### Interpretability & Fairness Audit
                    Grad-CAM explanations extracted from `layer4[-1]` of ResNet18 across test cohorts.
                    """
                )
                if os.path.exists("out/error_analysis/subgroup_auroc.png"):
                    gr.Image("out/error_analysis/subgroup_auroc.png", label="Subgroup Performance Across Biological Sex, Age Brackets, and Anatomical Sites")
                gr.Markdown(
                    r"""
                    **Subgroup Insights:**
                    - **Gender Parity:** Male AUROC = **0.8913** ($N=233$) vs Female AUROC = **0.9014** ($N=155$). Negligible disparity ($\Delta = 0.0101$).
                    - **Anatomical Site Alert:** Facial skin exhibits reduced specificity (**35.71%**) due to chronic actinic elastosis and solar lentigines. Clinicians should expect higher false-positive rates on facial lesions.
                    - **False Negative Root Causes ($N=9$):** Amelanotic/hypopigmented lesions, heavy hair occlusion, and in-situ lesions with low boundary contrast.
                    """
                )

    return demo

def launch(port=7860, share=False):
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=port, share=share)

if __name__ == "__main__":
    launch()
