# Refactor: Organize ipomdp_shielding Project

## Context

The codex run added Carr shield comparison and perception realization optimization to the RL shielding experiments, but put files in `~/claude/` instead of the project directory. You moved things into `~/claude/ipomdp_shielding_/` but the structure is messy: loose scripts at root, results scattered across 3 locations, docs buried inside the package, config paths pointing to `/tmp/` and `./data/`, and shell scripts buried inside the package.

**Goal**: Clean project layout where it's obvious how to run experiments and where results go.

## Proposed Structure

```
ipomdp_shielding_/                      # Project root
├── ipomdp_shielding/                   # Python package (source)
│   ├── Models/                          # (unchanged)
│   ├── Propagators/                     # (unchanged)
│   ├── Evaluation/                      # (unchanged)
│   ├── MonteCarlo/                      # (unchanged)
│   ├── CaseStudies/                     # (unchanged)
│   └── experiments/                     # Stays in package (relative imports)
│       ├── configs/                     # All experiment configs
│       ├── sweeps/                      # Parameter sweeps
│       ├── run_rl_shield_experiment.py
│       ├── run_coarse_experiment.py
│       ├── run_carr_comparison.py
│       └── ...
├── docs/                                # MOVED from ipomdp_shielding/docs/
├── scripts/                             # NEW: collected loose scripts
│   ├── debug_lifted_detailed.py
│   ├── debug_lifted_shield.py
│   ├── run_perfect_obs_experiment.py
│   ├── run_perfect_obs_fixed.py
│   ├── test_constrained_initial_support.py
│   └── test_perfect_observations.py
├── results/                             # Consolidated output directory
│   ├── prelim/                          # Prelim experiment outputs + figures
│   ├── full/                            # Full experiment outputs + figures
│   ├── carr_comparison/                 # Carr comparison outputs
│   └── cache/                           # RL agent + realization caches
├── models/                              # Trained model weights (keep as-is)
├── run_experiments.sh                   # NEW: single entry point script
├── pyproject.toml
├── CARR_IMPLEMENTATION_SUMMARY.md
├── IMPLEMENTATION_SUMMARY.md
└── README.md                            # NEW: clear instructions
```

## Key Design Decisions

1. **experiments/ stays inside the package** - It uses deep relative imports (`from ...CaseStudies.Taxinet`) across 20+ files. Moving it out would require rewriting 40+ import lines for no functional benefit.

2. **docs/ moves to project root** - Not Python code, shouldn't be in installable package.

3. **Single `results/` directory** - Currently scattered across `~/claude/results/`, `~/claude/data/`, and `ipomdp_shielding_/images/`. Consolidate under `results/`.

4. **Cache paths out of /tmp/** - Config files point RL caches to `/tmp/` which gets cleared on reboot. Move to `results/cache/`.

5. **Shell scripts at project root** - Currently buried inside `ipomdp_shielding/experiments/`. Move to root for discoverability.

## Steps

### Phase 1: Create directory structure and move files

1. Create `docs/`, `scripts/`, `results/{prelim,full,carr_comparison,cache}/`
2. Move `ipomdp_shielding/docs/*` → `docs/`
3. Move 6 loose root scripts → `scripts/`
4. Move existing results from `~/claude/results/*.json` → `results/carr_comparison/`
5. Move existing results from `~/claude/data/prelim/` → `results/prelim/`
6. Move `images/` contents → `results/` (old visualizations)
7. Remove empty `ipomdp_shielding/docs/` directory
8. Remove old `ipomdp_shielding/experiments/prelim.sh` and `full.sh`
9. Remove redundant experiment markdown files (README.md, EXPERIMENTS_README.md, MODULARIZATION_SUMMARY.md, QUICK_REFERENCE.md, plan.txt, README_CARR_COMPARISON.md) - their content will be captured in the new README.md

### Phase 2: Update config output paths (9 files)

All 8 experiment configs + carr_comparison_config: change `./data/` → `results/` and `/tmp/` → `results/cache/`

**Files to update:**
- `configs/rl_shield_taxinet_prelim.py` - paths → `results/prelim/`, `results/cache/`
- `configs/rl_shield_taxinet_full.py` - paths → `results/full/`, `results/cache/`
- `configs/rl_shield_cartpole_prelim.py` - same pattern
- `configs/rl_shield_cartpole_full.py` - same pattern
- `configs/coarse_taxinet_prelim.py` - paths → `results/prelim/`
- `configs/coarse_taxinet_full.py` - paths → `results/full/`
- `configs/coarse_cartpole_prelim.py` - same pattern
- `configs/coarse_cartpole_full.py` - same pattern
- `configs/carr_comparison_config.py` - paths → `results/carr_comparison/`

### Phase 3: Fix Carr comparison imports

Convert absolute → relative imports in 3 files for consistency:
- `carr_comparison_experiment.py`
- `run_carr_comparison.py`
- `optimal_realization_example.py` (if it uses absolute imports)

### Phase 4: Create run_experiments.sh at project root

Single entry-point script that replaces `prelim.sh`/`full.sh`:
```bash
./run_experiments.sh prelim    # Run all prelim experiments
./run_experiments.sh full      # Run all full experiments
./run_experiments.sh carr      # Run Carr comparison
```

### Phase 5: Create README.md

Clear instructions:
- What this project is
- How to install (`pip install -e .`)
- How to run experiments (the 3 main experiment types)
- Where results go
- Package structure overview

### Phase 6: Add .gitignore and clean up

- Add proper `.gitignore` (results/, __pycache__/, *.egg-info, models/*.pt)
- Remove `ipomdp_shielding/README_SETUP.md` and `ipomdp_shielding/setup.sh` if redundant

## Verification

1. `pip install -e .` still works
2. `python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim` runs successfully
3. `python -m ipomdp_shielding.experiments.run_carr_comparison` runs successfully
4. Results appear in `results/` directory
5. All imports resolve correctly
