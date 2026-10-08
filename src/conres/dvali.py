"""Community memberships from Dvali, Seguin, Betzel & Leifer (2025), PRX Life 3, 033021.

First hierarchical level of the hierarchical SBM, transcribed from the paper's
Table I (anatomical connectome, 20 communities) and Table II (signal propagation
network, 6 communities). Both cover the same 188 head neurons.

The signaling communities were fit to the Randi et al. (2023) atlas, the same
data the path operator is scored against, so their predictive scores are not
out-of-sample.
"""
from __future__ import annotations

import numpy as np

ANATOMY = {
    "1": "RIH",
    "2": "RIBL RIBR",
    "3": "RIAL RIAR",
    "4": "AIBL AIBR RIML RIMR",
    "5": "URYDL URYDR URYVL URYVR",
    "6": "AVEL AVER RIGL RIGR RIS RMFL",
    "7": "AVKL AVKR SAADL SAADR SAAVL SAAVR",
    "8": "AVAL AVAR AVBL AVBR AVDL AVDR AVL RID",
    "9": "ADAL ADAR ALA AQR AVG AVJL AVJR FLPL FLPR",
    "10": "RIVL RIVR RMDL RMDR RMFR SIAVR SMDDL SMDDR SMDVL SMDVR",
    "11": "ADEL ADER RICL RICR RMGL RMGR URBL URBR URXL URXR",
    "12": "ADFL ADFR AIZL AIZR AUAL AUAR AWBL AWBR BAGL BAGR RIR",
    "13": "SIADL SIADR SIAVL SIBDL SIBDR SIBVL SIBVR SMBDL SMBDR SMBVL SMBVR",
    "14": "IL1L IL1R OLLL OLLR RMDDL RMDDR RMDVL RMDVR RMED RMEL RMER RMEV",
    "15": "IL1DL IL1DR IL1VL IL1VR IL2VL IL2VR RIPL RIPR URADL URADR URAVL URAVR",
    "16": "AS1 DA1 DB1 DB2 DD1 SABD SABVL SABVR VA1 VB1 VB2 VD1 VD2",
    "17": "CEPDL CEPDR CEPVL CEPVR IL2DL IL2DR IL2L IL2R OLQDL OLQDR OLQVL OLQVR RMHL",
    "18": "ADLL ADLR AIML AIMR ASHL ASHR ASJL ASJR ASKL ASKR AVFL AVFR AVHL AVHR RIFL RIFR",
    "19": "AFDL AFDR AIAL AIAR AINL AINR AIYL AIYR ASEL ASER ASGL ASGR ASIL ASIR AWAL AWAR "
          "AWCL AWCR",
    "20": "I1L I1R I2L I2R I3 I4 I5 I6 M1 M2L M2R M3L M3R M4 M5 MCL MCR MI NSML NSMR",
}

SIGNALING = {
    "I": "I2L I2R I3 I4 IL2R M1 M2L M2R M3L M3R M4 MCL MCR MI NSML NSMR URYVR",
    "II": "AVL BAGL BAGR CEPVL I1L I1R I6 IL1DL IL1DR IL1VL IL1VR IL2DR IL2VR M5 OLLL OLLR "
          "OLQDR OLQVR RIPR RMER SAADL URBR URYDL URYVL",
    "III": "AFDL AIML AIMR AIYL AIYR ASJL AUAL AUAR AVBL RICL RIH RIML RIMR RMDDR RMDL RMFL "
           "RMFR RMGL SABVL SABVR SIBVR SMBDL SMBVL SMBVR SMDDL SMDDR VB2",
    "IV": "ADAL ADAR ADEL ADER AIBR AINL AIZL AIZR AQR AS1 AVFL AVFR AVG AVKL AVKR DB1 DB2 "
          "DD1 FLPL FLPR RIBL RIBR RICR RIFR RIGL RIGR RIS RMHL SABD SIBVL SMBDR URBL VA1 VB1",
    "V": "ADFR AIAL AIAR AIBL ALA AWCL CEPVR DA1 I5 IL1L IL1R IL2DL IL2L IL2VL OLQDL OLQVL "
         "RID RIFL RIPL RIR RMDDL RMED RMEL RMEV RMGR SAADR SIADL SIADR SIAVL SIAVR SIBDL "
         "SIBDR URADL URADR URAVL URAVR URYDR VD1 VD2",
    "VI": "ADFL ADLL ADLR AFDR AINR ASEL ASER ASGL ASGR ASHL ASHR ASIL ASIR ASJR ASKL ASKR "
          "AVAL AVAR AVBR AVDL AVDR AVEL AVER AVHL AVHR AVJL AVJR AWAL AWAR AWBL AWBR AWCR "
          "CEPDL CEPDR RIAL RIAR RIVL RIVR RMDR RMDVL RMDVR SAAVL SAAVR SMDVL SMDVR URXL URXR",
}

PHARYNX = ANATOMY["20"].split()


def membership(table: str) -> dict[str, str]:
    """Neuron name -> community label for "anatomy" or "signaling"."""
    src = {"anatomy": ANATOMY, "signaling": SIGNALING}[table]
    out = {}
    for c, names in src.items():
        for n in names.split():
            assert n not in out, f"{n} listed twice in {table}"
            out[n] = c
    return out


def labels(nodes: list[str], table: str) -> np.ndarray:
    """Integer labels over `nodes`, -1 for neurons outside the 188."""
    m = membership(table)
    keys = sorted(set(m.values()), key=lambda s: (len(s), s))
    code = {c: i for i, c in enumerate(keys)}
    return np.array([code[m[n]] if n in m else -1 for n in nodes])
