#!/usr/bin/env python3
import os
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import argparse
import ROOT
from array import array

parser = argparse.ArgumentParser(description="ROOT prediction and merge script with PyROOT output")
parser.add_argument("-i", "--input", required=True, help="Path to input ROOT file")
parser.add_argument("-o", "--output", required=True, help="Path to output ROOT file")
args = parser.parse_args()

input_file = args.input
output_file = args.output
tree_name = "miniTree"

model_path = "testModel_v1.hdf5"
if not os.path.isfile(model_path) or not model_path.endswith(".hdf5"):
    raise ValueError(f"Model file {model_path} not found or not a .hdf5 file")
model = load_model(model_path, compile=False)

scaler_dict = joblib.load("trained_scaler.pkl")
scaler = scaler_dict["scaler"] if isinstance(scaler_dict, dict) else scaler_dict

with uproot.open(input_file) as f:
    if tree_name not in f:
        raise RuntimeError(f"Tree '{tree_name}' not found in {input_file}")
    arrays = f[tree_name].arrays(library="ak")
    # arrays = f[tree_name].arrays(library="ak", entry_start=0, entry_stop=50000)

event_vars = ["event", "run", "hf_en", "nvtx", "nele", "nhf", "nmc"]
hf_vars = ["hf_eta", "hf_pt", "hf_phi", "hf_iso", "hf_ecal", "hf_hcal"]
scale_cols = [
    "nvtx", "nele", "nhf", "nmc",
    "hf_en", "hf_pt", "hf_eta", "hf_phi",
    "hf_iso", "hf_ecal", "hf_hcal",
    "idx", "evtwt"
]
model_input_columns = ["hf_eta", "hf_pt", "hf_phi", "hf_iso", "hf_ecal", "hf_hcal"]

slim_df = pd.DataFrame()

ref_branch = "hf_en"
for var in event_vars:
    broadcasted = ak.broadcast_arrays(arrays[var], arrays[ref_branch])[0]
    slim_df[var] = ak.flatten(broadcasted, axis=1)
for var in hf_vars:
    slim_df[var] = ak.flatten(arrays[var], axis=1)

event_indices = np.arange(len(arrays["event"]))
broadcasted_idx = ak.broadcast_arrays(np.array(event_indices), arrays[ref_branch])[0]
slim_df["idx"] = ak.flatten(broadcasted_idx, axis=1)
slim_df["evtwt"] = np.ones(len(slim_df))

missing_cols = [col for col in scale_cols if col not in slim_df.columns]
if missing_cols:
    raise RuntimeError(f"Missing columns from DataFrame: {missing_cols}")

scaled_values = scaler.transform(slim_df[scale_cols].values)
scaled_df = slim_df.copy()
scaled_df[scale_cols] = scaled_values

X_infer = scaled_df[model_input_columns].values
predictions = model.predict(X_infer, verbose=1)
flat_prob_sig = predictions[:, 0]

candidate_counts = np.array(ak.to_list(arrays["nhf"]))
if flat_prob_sig.shape[0] != np.sum(candidate_counts):
    raise ValueError("Mismatch: total number of predictions does not equal sum of nhf counts.")

nested_prob_sig = ak.unflatten(flat_prob_sig, candidate_counts)
prob_sig_list = ak.to_list(nested_prob_sig)
event_list = ak.to_list(arrays["event"])

event_prob_map = {evt: prob for evt, prob in zip(event_list, prob_sig_list)}

f_orig = ROOT.TFile.Open(input_file, "READ")
t_orig = f_orig.Get(tree_name)

f_new = ROOT.TFile(output_file, "RECREATE")
t_new = t_orig.CloneTree(0)

prob_sig_vec = ROOT.std.vector('float')()
t_new.Branch("prob_sig", prob_sig_vec)

n_entries = t_orig.GetEntries()
for i in range(n_entries):
    t_orig.GetEntry(i)
    evt = t_orig.event
    prob_values = event_prob_map.get(evt, [])
    prob_sig_vec.clear()
    for val in prob_values:
        prob_sig_vec.push_back(float(val))
    t_new.Fill()

t_new.Write()
f_new.Close()
f_orig.Close()

print(f"Merged file with original branches and 'prob_sig' branch saved to '{output_file}'.")
