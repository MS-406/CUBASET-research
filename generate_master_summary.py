import os
import pandas as pd

def generate_master_summary():
    mission_csv = "generalization_v2/results/tables/master_sota_generalization_benchmark.csv"
    ext_csv = "external_benchmark/results/external_generalization_benchmark.csv"
    
    df_m = pd.read_csv(mission_csv)
    df_e = pd.read_csv(ext_csv)

    # Extract external scores per model
    smd_raw = {}
    smd_aff = {}
    nab_raw = {}
    nab_aff = {}

    for _, row in df_e.iterrows():
        m_name = row["Model / Method"]
        proto = row["Dataset Protocol"]
        if "SMD" in proto:
            smd_raw[m_name] = row["Raw-F1 (Primary)"]
            smd_aff[m_name] = row["Affiliation-F1"]
        elif "NAB" in proto:
            nab_raw[m_name] = row["Raw-F1 (Primary)"]
            nab_aff[m_name] = row["Affiliation-F1"]

    # Match model names
    name_map = {
        "v1 (Baseline ConvAE)": "v1 (Baseline ConvAE)",
        "v2 USAD Teacher (Dual-AE)": "v2 USAD Teacher (Dual-AE)",
        "v2 USAD + CORAL Domain Adaptation": "v2 USAD + CORAL Domain Adaptation",
        "v2 Anomaly Transformer (Assoc. Discrepancy)": "v2 Anomaly Transformer",
        "v2 PatchTST Multi-Scale Backbone": "v2 PatchTST Multi-Scale Backbone",
        "v2 Distilled Edge Student (Proposed)": "v2 Distilled Edge Student (Proposed)"
    }

    summary_rows = []
    for _, row in df_m.iterrows():
        orig_name = row["Model / Method"]
        ext_key = name_map.get(orig_name, orig_name)

        summary_rows.append({
            "Model / Architecture": orig_name,
            "Params": row["Params"],
            "Footprint": row["Footprint"],
            "NASA Raw-F1": row["NASA Raw-F1"],
            "NASA Aff-F1": row["NASA Aff-F1"],
            "OPS-SAT Raw-F1": row["OPS-SAT Raw-F1"],
            "SMD Raw-F1": smd_raw.get(ext_key, 0.0),
            "NAB Raw-F1": nab_raw.get(ext_key, 0.0),
            "NAB Aff-F1": nab_aff.get(ext_key, 0.0),
            "Type / Role": row["Type"]
        })

    master_df = pd.DataFrame(summary_rows)
    
    out_csv = "results/tables/final_master_research_summary.csv"
    master_df.to_csv(out_csv, index=False)
    
    out_tex = "results/tables/final_master_research_summary.tex"
    master_df.to_latex(
        out_tex, 
        index=False, 
        caption="Consolidated cross-domain anomaly detection benchmark summarizing aerospace in-domain (NASA SMAP/MSL), aerospace cross-mission (ESA OPS-SAT-AD), and out-of-domain industrial transfer (SMD and NAB) under the un-gamed Raw-F1 protocol."
    )

    print("=" * 80)
    print("      CONSOLIDATED MASTER RESEARCH SUMMARY TABLE")
    print("=" * 80)
    print(master_df.to_string(index=False))
    print(f"\n[Saved Master CSV] {out_csv}")
    print(f"[Saved Master TeX] {out_tex}")

if __name__ == "__main__":
    generate_master_summary()
