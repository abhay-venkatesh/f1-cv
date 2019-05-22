import simplejson as json
from pathlib import Path


def fix_map(coco_evals):
    coco_evals_ = []
    for coco_eval in coco_evals:
        coco_eval_ = coco_eval
        if coco_eval["category_id"] == 0:
            coco_eval_["category_id"] = 183
        else:
            coco_eval_["category_id"] += 91
        coco_evals_.append(coco_eval_)
    return coco_evals_


if __name__ == "__main__":
    coco_evals = []
    for i in range(0, 10):
        filepath = Path("coco_stuff_deep_lab_val_" + str(i), "outputs",
                        "coco_result.json")
        with open(filepath) as f:
            coco_evals.extend(json.load(f))

    with open("coco_eval.json", "w+") as f:
        json.dump(coco_evals, f)

    coco_evals_ = fix_map(coco_evals)
    with open("coco_eval_.json", "w+") as f:
        json.dump(coco_evals_, f)
