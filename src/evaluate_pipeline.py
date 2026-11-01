# evaluate_pipeline.py
# Orchestrates Stages 1 through 6 and outputs per-category flow results
sanitizer = ForensicDataSanitizer(
    ablation_level="Set_C", label_col="Attack", binary_col="Label"
)
df_clean = sanitizer.clean_and_harmonize(raw_df, is_train_reference=True)

X_all, y_all = sanitizer.encode_categories(
    df_clean, zero_day_families=["Backdoor_ZeroDay"]
)
split, scaler = create_temporal_four_way_split(
    X_all, y_all, ratios=(0.55, 0.15, 0.15, 0.15)
)

backbone = HybridNIDSBackbone(
    feature_names=sanitizer.feature_columns_, random_state=42
)
backbone.fit(split.X_train, split.y_train)

V_train, _ = backbone.extract_penultimate_embeddings(split.X_train)
V_cal, _ = backbone.extract_penultimate_embeddings(split.X_cal)
V_test, logits_test = backbone.extract_penultimate_embeddings(split.X_test)

y_pred_train = np.argmax(backbone.predict_proba(split.X_train), axis=1)
osr_detector = DualStageOpenSetDetector(
    tail_size_eta=50, attenuation_vartheta=0.85, rec_quantile_theta=0.96
)
osr_detector.fit(V_train, split.y_train, y_pred_train, V_cal, split.y_cal)

probs_cal = backbone.predict_proba(split.X_cal)
probs_test = backbone.predict_proba(split.X_test)

mondrian_cp = MondrianConformalCalibrator(alpha=0.10, random_state=42)
mondrian_cp.fit(probs_cal, split.y_cal)

pred_set_mask, evidence_margin, p_values = mondrian_cp.predict_sets_and_margins(
    probs_test
)
zero_day_prob, is_zero_day_rejected = osr_detector.score_open_set(
    V_test, logits_test
)

triage = SOCTriageEngine(
    inverse_category_map=sanitizer.inverse_category_map_, tau_hi=0.25
)
flow_df, category_summary, global_summary = triage.route_and_evaluate(
    probs_test,
    pred_set_mask,
    evidence_margin,
    zero_day_prob,
    is_zero_day_rejected,
    split.y_test,
)