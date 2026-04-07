from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    """
    COCO-style dataset with GT-level text
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 🔥 여기에서 필터링 수행
        valid_inds = self._filter_imgs()

        print(f"[Filter] before: {len(self.data_infos)}")
        self.data_infos = [self.data_infos[i] for i in valid_inds]
        print(f"[Filter] after: {len(self.data_infos)}")


    def get_ann_info(self, idx):
        ann_info = super().get_ann_info(idx)

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        texts = []
        for ann in anns:
            if 'text' not in ann:
                raise KeyError('Annotation missing "text" field')
            texts.append(ann['text'])

        ann_info['texts'] = texts
        
        return ann_info

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _filter_imgs(self, min_size=32):
        """Filter images without valid texts"""
        valid_inds = []
        
        for i, img_info in enumerate(self.data_infos):
            ann = self.get_ann_info(i)
            texts = ann.get('texts', [])

            if hasattr(texts, 'data'):
                texts = texts.data

            if len(texts) > 0:
                valid_inds.append(i)

        return valid_inds
