# CardioVision 3D: Diagnostic Assessment & Multi-Vessel Spatial Risk Twin

## Executive Overview
CardioVision 3D is a clinical decision-support platform designed to predict overall Coronary Artery Disease (CAD) and vessel-specific stenosis across the three major coronary arteries: Left Anterior Descending (LAD), Left Circumflex (LCX), and Right Coronary Artery (RCA). Clinical tabular measurements are mapped directly into an interactive 3D cardiac digital twin with real-time SHAP explainability.

---

## 1. Predictive Pipeline & Leakage Mitigation
- **Dataset:** UCI Extension of Z-Alizadeh Sani Dataset (303 patient records, 55 clinical parameters).
- **Target Leakage Prevention:** Adhering strictly to clinical guidelines, the ground-truth targets (`LAD`, `LCX`, `RCA`, `CAD`) and catheterization markers (`Cath`) were isolated and excluded from the feature training matrix.
- **Model Architecture:** Multi-target XGBoost ensemble classifiers with class-balanced weighting (`scale_pos_weight`) to ensure robust sensitivity across multi-vessel stenosis.
- **Validation Metrics:**
  - CAD Overall: ~82.0% Accuracy | ROC-AUC ~0.84
  - LAD Stenosis: ~70.5% Accuracy | ROC-AUC ~0.71
  - LCX Stenosis: ~64.0% Accuracy | ROC-AUC ~0.67
  - RCA Stenosis: ~70.5% Accuracy | ROC-AUC ~0.66

---

## 2. Multimodal 3D Anatomical Twin
- **WebGL Interactive Viewport:** Renders the myocardium silhouette with dynamic spatial arteries.
- **Dynamic Color Interpolation:** 
  - Emerald Green (`#10B981`): Low stenosis risk (< 40%)
  - Amber Warning (`#F59E0B`): Borderline stenosis (40% - 64%)
  - Crimson Red (`#EF4444`): Critical stenosis (≥ 65%)
- **Spatial Exploration:** 360° orbital rotation, panning, and anatomical segment tooltips linking stenosis scores to perfusion regions.

---

## 3. Explainable AI (SHAP Framework)
- Integrated TreeExplainer calculating exact log-odds attributions for physiological variables.
- Direct visualization of leading risk contributors, including Typical Chest Pain, Ejection Fraction (EF-TTE), Regional Wall Motion Abnormalities (RWMA), and blood lipid profiles.

---

## 4. Regulatory & Safety Guardrail
- Configured strictly as a Software-as-a-Medical-Device (SaMD) Class IIa decision-support tool.
- Non-diagnostic disclaimer integrated into the main interface to mandate licensed clinical review.