import torch
from mmdet.registry import MODELS              # 변경점: DETECTORS → MODELS
                                               # 2.x: mmdet.models.DETECTORS
                                               # 3.x: mmdet.registry.MODELS
from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.structures import DetDataSample     # 변경점: 3.x 전용 데이터 구조
from mmdet.utils import OptSampleList


@MODELS.register_module()
class CustomMaskRCNN(MaskRCNN):
    """
    Mask R-CNN variant that always takes text input
    (text affects backbone/neck features only, not losses)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def extract_feat(self, batch_inputs: torch.Tensor,
                     batch_data_samples: OptSampleList = None):
        """
        batch_inputs      : Tensor (B, C, H, W)
        batch_data_samples: list[DetDataSample], each contains metainfo['texts']
        """

        if batch_data_samples is not None:
            texts = [
                ds.metainfo['texts']
                for ds in batch_data_samples
            ]
        else:
            texts = None

        x = self.backbone(batch_inputs, gt_texts=texts)

        if self.with_neck:
            if hasattr(self.backbone, 'text_embeddings'):
                text_embeddings = self.backbone.text_embeddings
                if hasattr(self.neck, 'set_text_embeddings'):
                    self.neck.set_text_embeddings(text_embeddings)
            x = self.neck(x)

        return x

    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: list) -> dict:
        """
        학습 시 호출. losses dict 반환.
        변경점: forward_train(img, img_metas, gt_bboxes, ...) → loss(batch_inputs, batch_data_samples)
        gt_bboxes, gt_labels, gt_masks 등은 batch_data_samples 안에 포함됨.
        """
        x = self.extract_feat(batch_inputs, batch_data_samples)

        losses = dict()

        # RPN
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = batch_data_samples
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                ds.proposals for ds in batch_data_samples
            ]

        # ROI
        roi_losses = self.roi_head.loss(
            x, rpn_results_list, batch_data_samples
        )
        losses.update(roi_losses)

        return losses

    def predict(self, batch_inputs: torch.Tensor,
                batch_data_samples: list,
                rescale: bool = True) -> list:
        """
        추론 시 호출. list[DetDataSample] 반환.
        변경점: simple_test(imgs, img_metas, ...) → predict(batch_inputs, batch_data_samples)
        """
        x = self.extract_feat(batch_inputs, batch_data_samples)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples,
                                                     rescale=False)
        else:
            rpn_results_list = [
                ds.proposals for ds in batch_data_samples
            ]

        results = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale
        )
        return results

    def _forward(self, batch_inputs: torch.Tensor,
                 batch_data_samples: OptSampleList = None):
        """
        FLOPs 계산 등 dummy forward 용도.
        변경점: forward_dummy → _forward
        """
        B, _, H, W = batch_inputs.shape

        if batch_data_samples is None:
            # dummy data_samples 생성
            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            dummy_samples = []
            for _ in range(B):
                ds = DetDataSample()
                ds.set_metainfo(dict(
                    img_shape=(H, W),
                    ori_shape=(H, W),
                    pad_shape=(H, W),
                    scale_factor=(1.0, 1.0),
                    flip=False,
                    flip_direction=None,
                    img_path=None,
                    texts=['red organic apple'],
                ))
                dummy_samples.append(ds)
            batch_data_samples = dummy_samples

        x = self.extract_feat(batch_inputs, batch_data_samples)
        outs = ()

        if self.with_rpn:
            rpn_outs = self.rpn_head.forward(x)
            outs = outs + (rpn_outs,)

        return outs