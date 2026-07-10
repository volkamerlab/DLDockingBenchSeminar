import lmdb, pickle
import numpy as np

env = lmdb.open(
    "/home/bdldt_team007/DLDockingBenchSeminar/data/processed/train.lmdb",
    subdir=False, readonly=True, lock=False
)

with env.begin() as txn:
    for i in range(5):
        entry = pickle.loads(txn.get(f"{i}".encode()))
        conf = entry['coordinates'][0]
        holo = entry['holo_coordinates'][0]
        pocket = entry['pocket_coordinates'][0]
        print(f"\nSample {i} ({entry['pocket']}):")
        print(f"  conformer range: [{conf.min():.2f}, {conf.max():.2f}]")
        print(f"  holo range:      [{holo.min():.2f}, {holo.max():.2f}]")
        print(f"  pocket range:    [{pocket.min():.2f}, {pocket.max():.2f}]")
        print(f"  conformer center: {conf.mean(axis=0).round(2)}")
        print(f"  holo center:      {holo.mean(axis=0).round(2)}")
        print(f"  pocket center:    {pocket.mean(axis=0).round(2)}")
        # cross distance between holo and pocket
        cross = np.linalg.norm(
            holo[:, None, :] - pocket[None, :, :], axis=-1
        )
        print(f"  holo-pocket cross dist: min={cross.min():.2f}, max={cross.max():.2f}")
