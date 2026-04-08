from mmdet.registry import TRANSFORMS         
                                               
                                               


@TRANSFORMS.register_module()
class LoadTextAnnotations:


    def __init__(self):
        pass

    def __call__(self, results: dict) -> dict:
        texts = results.get('texts', None)

        if texts is None:
            
            ann_info = results.get('ann_info', {})
            texts = ann_info.get('texts', [])

        results['texts'] = texts

        return results