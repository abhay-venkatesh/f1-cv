import simplejson as json
from pathlib import Path


def fix_map(coco_evals):

    stuff_evals = []
    for coco_eval in coco_evals:
        coco_eval_ = coco_eval
        if coco_eval["category_id"] != 0:
            coco_eval_["category_id"] += 91
        stuff_evals.append(coco_eval_)

    with open(Path("panoptic_coco_categories.json")) as f:
        panoptic_cats = json.load(f)
    panoptic_cat_ids = [panoptic_cat["id"] for panoptic_cat in panoptic_cats]

    panoptic_stuff_vals = [
        stuff_eval for stuff_eval in stuff_evals
        if stuff_eval["category_id"] in panoptic_cat_ids
    ]

    return panoptic_stuff_vals


def resize(panoptic_stuff_vals):
    raise NotImplementedError
    with open(Path("panoptic_val2017.json")) as f:
        panoptic_val = json.load(f)


if __name__ == "__main__":
    coco_evals = []
    for i in range(0, 5):
        filepath = Path("coco_stuff_deep_lab_val_" + str(i), "outputs",
                        "coco_result.json")
        with open(filepath) as f:
            coco_evals.extend(json.load(f))

    panoptic_stuff_vals = fix_map(coco_evals)
    panoptic_stuff_vals = resize(panoptic_stuff_vals)

    with open("panoptic_stuff_vals.json", "w+") as f:
        json.dump(panoptic_stuff_vals, f)
