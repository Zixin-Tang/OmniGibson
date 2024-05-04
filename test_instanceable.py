from tqdm import tqdm

import omnigibson as og
from omnigibson.objects import USDObject
from omnigibson.scenes import Scene

INSTANCEABLE = False
NUM_ROBOTS = 100
asset_path = (
    "/home/svl/Downloads/ur5e/ur5e_instanceable.usd" if INSTANCEABLE else "/home/svl/Downloads/ur5e/ur5e_visuals.usd"
)

og.launch()

scene = Scene()
og.sim.import_scene(scene)

robots = []
for i in tqdm(range(NUM_ROBOTS)):
    robots.append(USDObject(f"robot_{i}", asset_path))
    og.sim.import_object(robots[-1])
    robots[-1].set_position([i * 2, 0, 0])

og.sim.play()
for i in range(100):
    og.sim.step()

og.shutdown()
