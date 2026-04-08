from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS        
                                            
                                            
@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    """
    COCO-style dataset with GT-level text
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        valid_inds = self._filter_imgs()

        print(f"[Filter] before: {len(self.data_list)}")
        self.data_list = [self.data_list[i] for i in valid_inds]
        print(f"[Filter] after: {len(self.data_list)}")
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """
        부모의 parse_data_info()를 호출한 뒤 'texts' 필드를 추가.
        """
        data_info = super().parse_data_info(raw_data_info)

        img_id = raw_data_info['raw_img_info']['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        texts = []
        for ann in anns:
            if 'text' not in ann:
                raise KeyError('Annotation missing "text" field')
            texts.append(ann['text'])

        data_info['texts'] = texts
        return data_info

    def _filter_imgs(self, min_size: int = 32) -> list:
        """texts가 존재하는 이미지만 유지"""
        valid_inds = []

        for i, data_info in enumerate(self.data_list):
            texts = data_info.get('texts', [])
            if len(texts) > 0:
                valid_inds.append(i)

        return valid_inds