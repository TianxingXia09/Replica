"""Run documented stages in dependency order; no data are downloaded."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'quick': [('-m', 'unittest', '-v'), ('minimax_benchmark.py',)],
    'core': [('run_phase2_benchmarks.py',), ('analyze_phase3_boundary.py',),
             ('run_phase3_mismatch.py',), ('run_readout_stress.py',),
             ('minimax_benchmark.py',), ('make_phase2_figures.py',),
             ('make_phase3_figures.py',), ('make_figures.py',), ('verify_phase3.py',)],
    'online': [('run_online_selector_pilot.py',), ('finalize_online_development.py',),
               ('run_online_selector_benchmark.py',), ('run_online_largeblock_benchmark.py',),
               ('make_phase4_figures.py',), ('verify_phase4.py',)],
    'validation': [('rebuild_validation.py',)],
    'early': [('run_falsification.py',), ('run_local_stress.py',)],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='quick')
    args = parser.parse_args()
    if args.stage == 'validation':
        required = [ROOT/'phase2_results/raw.csv',
                    ROOT/'phase3_results/boundary_selected_windows.csv']
        if any(not p.is_file() for p in required):
            parser.error('Run --stage core first; validation requires generated core outputs.')
    names = ['quick','core','online','validation','early'] if args.stage == 'all' else [args.stage]
    env = os.environ.copy()
    # Existing experiment scripts use up to four worker processes; avoid
    # multiplying those by a large BLAS thread count.
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        env[key] = '1'
    for name in names:
        for command in STAGES[name]:
            print('RUN',sys.executable,*command,flush=True)
            subprocess.run([sys.executable,*command],cwd=ROOT,env=env,check=True)

if __name__ == '__main__':
    main()
