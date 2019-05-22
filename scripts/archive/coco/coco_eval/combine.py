import simplejson as json
from pathlib import Path

if __name__ == "__main__":
    coco_evals = []
    for i in range(0, 10):
        filepath = Path("outputs", "coco_stuff_deep_lab_eval_" + str(i),
                        "outputs", "coco_result.json")
        with open(filepath) as f:
            coco_evals.extend(json.load(f))
    with open("coco_eval.json", "w+") as f:
        json.dump(coco_evals, f)
