from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools.cocoeval import COCOeval

data_dir = '../__DATASET/coco'
res_file = 'predict.json'

ann_type = 'bbox'
prefix = 'instances'
data_type = 'val2017'
ann_file = f'{data_dir}/annotations/{prefix}_{data_type}.json'

coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(res_file)

coco_eval = COCOeval(coco_gt, coco_dt, ann_type)

# Please be patient, it will take a long time to run.
# analyze() only needs to be run once,
# it will save data to precisions.npy by default.
coco_eval.analyze()

# makeplots() uses the data saved by analyze().
# Pictures are saved in the analyze_figures folder by default.
# Set the plots parameter according to your needs.
coco_eval.makeplots(plots=['overall', 'supercategory', 'category'])
