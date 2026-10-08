"""Time course of the evoked transient in the stimulated neuron itself.

For every stimulation event in the Randi 2023 wild-type recordings, the
stimulated neuron's trace is z-scored against the 10 volumes before onset
(at least 5 required; traces with any NaN skipped), and |z| is averaged at lags
0-79 volumes (2 Hz). The excess over the late baseline (mean |z| at lags >= 60)
is reported as a percentage of its peak, together with the share of the
integrated excess falling in the first 10, 20 and 40 volumes, the windows
removed by two_observables.py --excise.

Inputs: exported Randi 2023 recordings (conres.randi.EXPORTED/exported_data).
Outputs: stdout only.
"""
import sys, re, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from conres import randi
from conres.paths import tee_stdout                   # noqa: E402
tee_stdout(__file__)
folder = randi.EXPORTED/"exported_data"
LAG=80; acc=np.zeros(LAG); nacc=np.zeros(LAG)
for f in sorted(folder.glob("*_labels.txt")):
    i=int(re.match(r"(\d+)_labels",f.name).group(1))
    G=randi._gcamp(folder,i)
    sf=folder/f"{i}_stim_volume_i.txt"; nf=folder/f"{i}_stim_neurons.txt"
    if G is None or not sf.exists() or not nf.exists() or G.shape[0]<100: continue
    stims=[int(x) for x in sf.open() if x.strip()]
    sneu=[int(x) for x in nf.open() if x.strip()]
    for s,c in zip(stims,sneu):
        if not (0<=c<G.shape[1]): continue
        tr=G[:,c]
        if not np.isfinite(tr).all(): continue
        pre=tr[max(0,s-10):s]
        if pre.size<5: continue
        mu,sd=pre.mean(),max(pre.std(),1e-9)
        for L in range(LAG):
            t=s+L
            if t<len(tr): acc[L]+=abs((tr[t]-mu)/sd); nacc[L]+=1
p=acc/np.maximum(nacc,1); base=p[60:].mean(); ex=p-base; e0=max(ex[:6].max(),1e-9)
print("stimulated neuron, |z| relative to its own pre-stimulus baseline")
print("  lag(vol)  " + " ".join(f"{l:>5}" for l in (0,2,5,10,15,20,30,40,60,79)))
print("  |z|       " + " ".join(f"{p[l]:>5.2f}" for l in (0,2,5,10,15,20,30,40,60,79)))
print(f"  late baseline (lag>=60) = {base:.2f}")
print("  excess as % of peak excess:")
print("            " + " ".join(f"{100*ex[l]/e0:>4.0f}%" for l in (0,2,5,10,15,20,30,40,60,79)))
cum=np.cumsum(np.maximum(ex,0)); tot=cum[-1]
for w in (10,20,40):
    print(f"  excising {w:>2} volumes removes {100*cum[min(w,LAG-1)]/tot:.0f}% of the integrated evoked excess")
