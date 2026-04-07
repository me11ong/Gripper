import torch
from mmdet.models import DETECTORS
from mmdet.models.detectors.mask_rcnn import MaskRCNN


@DETECTORS.register_module()
class CustomMaskRCNN(MaskRCNN):
    """
    Mask R-CNN variant that always takes text input
    (text affects backbone features only, not losses)
    """

    def __init__(self, *args, **kwargs):
        super(CustomMaskRCNN, self).__init__(*args, **kwargs)

    # --------------------------------------------------
    # Feature extraction (text-aware)
    # --------------------------------------------------
    def extract_feat(self, img, img_metas):
        """
        img: Tensor (B, C, H, W)
        img_metas: list[dict], each contains 'texts'
        """
        assert img_metas is not None

        texts = [
            meta['texts'].data if hasattr(meta['texts'], 'data') else meta['texts']
            for meta in img_metas
        ]

        x = self.backbone(img, gt_texts=texts)

        # If using TextGuidedFPN, pass text embeddings to neck
        if self.with_neck:
            # Check if backbone has text_embeddings attribute (from GreedyViG_CostMatrix)
            if hasattr(self.backbone, 'text_embeddings'):
                text_embeddings = self.backbone.text_embeddings
                # Check if neck is TextGuidedFPN
                if hasattr(self.neck, 'set_text_embeddings'):
                    self.neck.set_text_embeddings(text_embeddings)
            x = self.neck(x)

        return x

    def forward_dummy(self, img):
        """
        Used for computing FLOPs.
        """

        batch_size, _, H, W = img.shape

        dummy_img_metas = [{
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': None,
            'img_norm_cfg': dict(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True
            ),
            'filename': None,
            'ori_filename': None,
            'texts': ['red organic apple']
        } for _ in range(batch_size)]

        x = self.extract_feat(img, dummy_img_metas)

        outs = ()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)

            proposal_list = self.rpn_head.get_bboxes(
                *rpn_outs,
                img_metas=dummy_img_metas,
                cfg=self.test_cfg.rpn
            )

        else:
            proposal_list = [torch.randn(1000, 4).to(img.device)
                            for _ in range(batch_size)]
            roi_outs = self.roi_head.forward_dummy(x, proposal_list)
            outs = outs + (roi_outs,)

        return outs


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        text is implicitly passed through img_metas
        """
        x = self.extract_feat(img, img_metas)

        losses = dict()


        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def simple_test(self, imgs, img_metas, proposals=None, rescale=False):
        """
        text is always used in extract_feat
        """
        x = self.extract_feat(imgs, img_metas)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(
                x, img_metas
            )
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x,
            proposal_list,
            img_metas,
            rescale=rescale
        )
  

    

