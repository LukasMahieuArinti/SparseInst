from detectron2.data.datasets import register_coco_instances
register_coco_instances("specprep11_train", {}, "data/20220404_specprep11_train.json", "data/images_20220404")
register_coco_instances("specprep11_val", {}, "data/20220404_specprep11_val.json", "data/images_20220404")