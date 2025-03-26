import os
import glob
import argparse
import ROOT

# cmssw = os.getenv("CMSSW_BASE") + "/src/"
cmssw  = "/afs/cern.ch/user/n/naislam/HFCaliberation/2024/test/CMSSW_14_0_15/src/"
workpath = "RadDam/HFmonitoring/Training/"
proxy = os.getenv("X509_USER_PROXY")

qcd_files = glob.glob("input_all.txt")

def is_root_file_valid(file_path):
    if not os.path.exists(file_path):
        return False

    try:
        root_file = ROOT.TFile.Open(file_path, "READ")
        if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
            root_file.Close()
            return False

        keys = root_file.GetListOfKeys()
        if not keys or keys.GetSize() == 0:
            root_file.Close()
            return False

        if not root_file.Get("miniTree"):
            root_file.Close()
            return False

        root_file.Close()
        return True
    except Exception:
        return False

for qcd_file in qcd_files:
    cuts = qcd_file.replace(".txt", "")
    storage_path = f"/eos/user/n/naislam/HFCalibration/2024/2024H/Untuplizer/output_final/{cuts}/"
    
    with open(qcd_file, "r") as f:
        root_files = [line.strip() for line in f.readlines() if line.strip()]

    total_jobs = len(root_files)
    os.makedirs(f"jobs_{cuts}", exist_ok=True)
    os.makedirs(storage_path, exist_ok=True)

    script_filename = f"jobs_{cuts}/condorjob.sh"
    condor_filename = f"jobs_{cuts}/condorjob.sub"
    valid_jobs = []

    with open(script_filename, "w") as sh_file:
        sh_file.write("#!/bin/sh\n")
        sh_file.write(f"cd {cmssw}\n")
        sh_file.write(f"export X509_USER_PROXY={proxy}\n")
        sh_file.write("source /cvmfs/cms.cern.ch/cmsset_default.sh\n")
        sh_file.write("export SCRAM_ARCH=el9_amd64_gcc11\n")
        sh_file.write("eval `scramv1 runtime -sh`\n")
        sh_file.write(f"cd {cmssw}{workpath}\n\n")

        for i, root_file in enumerate(root_files):
            output_file = f"{storage_path}output_file_{i}.root"

            if is_root_file_valid(output_file):
                continue

            valid_jobs.append(i)

            sh_file.write(f"if [ $1 -eq {i} ]; then\n")
            sh_file.write(f"  echo 'Running predict.py on {root_file}'\n")
            sh_file.write(f"  python3 predict.py -i {root_file} -o {output_file}\n")
            # sh_file.write(f"  python3 Predictor_final.py -i {root_file} -o {output_file}\n")
            sh_file.write("fi\n")

    with open(condor_filename, "w") as condor_file:
        condor_file.write(f"executable = {cmssw}{workpath}jobs_{cuts}/condorjob.sh\n")
        condor_file.write("arguments  = $(ProcId)\n")
        condor_file.write(f"output     = {cmssw}{workpath}jobs_{cuts}/condorjob.$(ClusterId).$(ProcId).out\n")
        condor_file.write(f"error      = {cmssw}{workpath}jobs_{cuts}/condorjob.$(ClusterId).$(ProcId).err\n")
        condor_file.write(f"log        = {cmssw}{workpath}jobs_{cuts}/condorjob.$(ClusterId).log\n")
        condor_file.write("+JobFlavour = \"nextweek\"\n")
        condor_file.write("RequestCpus = 2\n")  
        condor_file.write("request_memory = 2GB\n") 
        condor_file.write("request_disk = 5GB\n")  
        condor_file.write("on_exit_hold = (ExitBySignal == True) || (ExitCode != 0) || (!(ExitSignal =?= UNDEFINED))\n")
        condor_file.write("periodic_release = (NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 600)\n")
        condor_file.write(f"queue {len(valid_jobs)}\n")

    valid_output_count = sum(1 for i in range(total_jobs) if is_root_file_valid(f"{storage_path}output_file_{i}.root"))

    if len(valid_jobs) == 0:
        print("No jobs need to be submitted. All output files are valid.")
    else:
        print(f"Condor submission file created: {condor_filename}")
        print(f"Run: condor_submit {condor_filename}")
        os.system(f"condor_submit {condor_filename}")

    if valid_output_count == total_jobs:
        print(f"✅ All {total_jobs} output files are valid and match the input text file!")
    else:
        print(f"⚠️ WARNING: Expected {total_jobs} output files but found {valid_output_count} valid ones!")
        print("Some jobs may not have completed correctly. Consider rechecking failed jobs.")

