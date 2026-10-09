   

UCM_AID_LABELS = [
    "airplane",
    "bare soil",
    "building",
    "car",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tank",
    "tree",
    "water",
]

DFC_LABELS = [
    "impervious surface",
    "water",
    "clutter",
    "vegetation",
    "building",
    "tree",
    "boat",
    "car",
]

LABEL_SETS = {
    "ml_ucm": UCM_AID_LABELS,
    "ml_aid": UCM_AID_LABELS,
    "ml_dfc": DFC_LABELS,
}

                                                     
                                                            
NUM_SCENES = {"ml_ucm": 10, "ml_aid": 10, "ml_dfc": 6}

                                                                          
ALPHA_TYPICAL = 0.5
