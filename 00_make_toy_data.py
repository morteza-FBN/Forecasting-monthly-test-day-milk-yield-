from pathlib import Path
import numpy as np
import pandas as pd

Path("example_data").mkdir(exist_ok=True)
rng = np.random.default_rng(42)

def make_toy(n=90, external=False):
    rows=[]
    for idx in range(n):
        cow = 100000+idx if not external else 200000+idx
        parity = int(rng.choice([1,2,3,4], p=[0.45,0.25,0.18,0.12]))
        peak = 42 + 2.5*parity + rng.normal(0, 4) - (5 if external else 0)
        dims = np.array([15,45,75,105,135,165,195,225,255]) + rng.normal(0, 3, 9)
        milk = [max(5, peak*(1-np.exp(-d/24))*np.exp(-0.0026*d) + rng.normal(0,1.5)) for d in dims]
        row = {"cow_id": cow, "parity": parity, "calving_date": "2015-01-01"}
        for i,v in enumerate(milk,1): row[f"milk_{i}"] = round(float(v),2)
        for i,v in enumerate(dims,1): row[f"dim_{i}"] = round(float(v),1)
        rows.append(row)
    return pd.DataFrame(rows)

dev = make_toy(90, False)
ext = make_toy(70, True)
dev.to_csv("example_data/toy_development_data.csv", index=False)
raw_ext = ext.rename(columns={"cow_id":"Dam_ID", "parity":"Dam_calving_parity", "calving_date":"Calving_Date"})
for i in range(1,10):
    raw_ext = raw_ext.rename(columns={f"milk_{i}":f"Milk_recorod_{i}", f"dim_{i}":f"Days_in_milk_{i}"})
raw_ext.to_csv("example_data/toy_external_raw.csv", index=False)
print("Wrote toy datasets to example_data/.")
