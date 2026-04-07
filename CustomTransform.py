from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC

@PIPELINES.register_module()

class LoadTextAnnotations:
    def __init__(self):
        pass

    def __call__(self, results):
        texts = results['ann_info']['texts']

        results['texts'] = DC(texts, cpu_only=True)

        return results


