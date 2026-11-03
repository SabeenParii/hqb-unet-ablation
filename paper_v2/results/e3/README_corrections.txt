E3 (P1_28r) run metadata corrections - recorded during training, before any analysis.

Scope: only run.json fields derived from history.csv (epochs_run, best_epoch, final_lr) can be wrong
when a Colab session stops before history.csv reaches Drive; Dice and all per-image results are
computed from the restored best weights and are not affected.

Audit 2026-11-01 07:15 UTC of all resumed runs finished so far (history.csv size vs epochs):
  E3v_f4_hc s4 (resumed 60)  history complete (95 epochs)
  E3v_f0_hf s0 (resumed 20)  complete (200)
  E3v_f1_hc s0 (resumed 20,60) complete (154)
  E3v_f3_ht s4 (resumed 80)  complete
  E3v_f1_hc s4 (resumed 160) complete (180)
  E3v_f1_hq s4 (resumed 180) INCOMPLETE -> corrected in e3v_run_corrections.csv

Analysis rule (fixed now): for every run with non-empty resumed_at, if the first epoch in history.csv
is > 0, epochs_run = last history epoch + 1 and best_epoch is taken from the resume-state snapshot
(e3v_state_snapshots) or from this file. Learning curves of such runs are reported as partial.
State snapshots of active runs are recorded at every monitoring check from now on.

Audit 2026-11-03 05:36 UTC (resumed runs finished since 2026-11-01):
  E3v_f3_hf s3 (resumed 120) complete
  E3v_f4_ht s1 (resumed 120) complete (overlap rows kept)
  E3v_f0_ht s3 (resumed 20)  complete (epochs 0-38 then 20-134; duplicates of 20-38 from the pre-resume process; last row per epoch used)
  E3v_f0_hq s3 (resumed 20)  INCOMPLETE (history starts at epoch index 20) -> corrected: epochs_run 140, best_epoch 130
                             (e3v_run_corrections_E3v_f0_hq_seed3.csv)
# 2026-11-03T14:04Z: E3 complete - last run.json E3v_f3_ht seed2 [B] 13:58 UTC (not resumed); no failed.json
