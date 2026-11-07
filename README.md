# A Decision-Centric Defense Framework for Cyber-Physical Systems

## Run
    pip install numpy scipy matplotlib python-docx
    python decision_iet_final.py          # experiments (about 15 min on 2 cores): results/*.json
    python build_paper.py                 # figures, LaTeX tables/text, response letter, title page
(`python build_paper.py --run` does both.)  Everything is deterministic (fixed seeds).

## Model summary
Jump LQ game with separate B and C and checked saddle-point conditions; cyber transition matrix induced by the strategies;
damage = Lambda*(1-exp(-eta*[increase of J*]^+)); attacker effort c(a); baseline = constant-kappa penalty;
all policies evaluated in the same game against the same 27 pure attackers + the uniform attacker; minimax-Q learning.