import os
import random
import numpy as np
from PIL import Image
import torch
import sys
sys.path.append("/mnt/E/Gripper/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingDeterministicWrapper:
    def __init__(self,
                 checkpoint_path,
                 device=None,
                 seed=42,
                 use_torch_deterministic=False,
                 verbose=True):
        """
        checkpoint_path: .pth 파일 경로
        device: 'cuda' 또는 'cpu' (None이면 자동)
        seed: random seed
        use_torch_deterministic: True로 하면 torch.use_deterministic_algorithms(True) 호출 (에러 발생 가능)
        """

        self._set_env_seed(seed, verbose)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        torch.use_deterministic_algorithms(False)
        
        if use_torch_deterministic:
            torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        self.model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
        state = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)

        self.verbose = verbose

    def _set_env_seed(self, seed, verbose=True):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"[deterministic wrapper] seed set to {seed}")

    def _prepare_input(self, img, expect_uint8=True):
        """
        img: 입력은 (H,W,3) numpy (uint8 or float 0-1) 또는 PIL Image 또는 torch Tensor (C,H,W)
        반환: numpy uint8 contiguous (H,W,3) — infer_image에 넣을 안전한 형태
        """
        # PIL -> numpy
        if isinstance(img, Image.Image):
            arr = np.array(img)
        elif isinstance(img, torch.Tensor):
            # (C,H,W) -> (H,W,C)
            arr = img.permute(1,2,0).cpu().numpy()
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            raise TypeError("img must be PIL.Image, torch.Tensor or numpy.ndarray")

        # ensure shape H,W,C
        if arr.ndim == 2:
            # grayscale to 3ch
            arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] not in (1,3):
            # maybe (C,H,W)
            if arr.shape[0] in (1,3):
                arr = np.transpose(arr, (1,2,0))
            else:
                raise ValueError("Unsupported image shape: %s" % (arr.shape,))

        # convert to uint8 (0..255) deterministically, without writing file
        if arr.dtype == np.uint8:
            out = arr.copy()
        else:
            # assume float in 0..1 or arbitrary scale; clip then convert
            arr = np.asarray(arr, dtype=np.float32)
            # If values appear to be 0..255 already, detect and handle
            vmin, vmax = arr.min(), arr.max()
            if vmin >= 0.0 and vmax <= 1.0:
                out = (arr * 255.0).round().astype(np.uint8)
            else:
                # if already 0..255 floats
                out = np.clip(arr, 0, 255).round().astype(np.uint8)

        # make contiguous C-order
        if not out.flags['C_CONTIGUOUS']:
            out = np.ascontiguousarray(out)

        return out

    def infer(self, img_np_hwc, return_numpy=True):
        """
        img_np_hwc: numpy array (H,W,3) or PIL or torch tensor
        return_numpy: True -> return numpy depth map, False -> return torch tensor on device
        """
        # 1) prepare input (no saving to disk)
        x = self._prepare_input(img_np_hwc, expect_uint8=True)  # uint8 HWC

        # 2) call model.infer_image (use same instance, model already on device)
        # ensure no_grad and deterministic environment
        with torch.no_grad():
            # pass numpy uint8 directly to infer_image (as in original usage)
            pred = self.model.infer_image(x)  # returns numpy array (H,W) or (H,W,1)
            # If the model returns torch tensor, convert; but typical infer_image returns numpy
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()

        # optionally normalize to 0..1 as float32 (consistent)
        pred = np.asarray(pred, dtype=np.float32)
        # ensure contiguous
        if not pred.flags['C_CONTIGUOUS']:
            pred = np.ascontiguousarray(pred)

        if return_numpy:
            return pred
        else:
            return torch.from_numpy(pred).to(self.device)

    # convenience utility: compare outputs deterministically
    def compare_infer(self, img_input, other_callable, atol=1e-6, rtol=1e-6):
        """
        img_input: input to pass to this wrapper
        other_callable: a callable that given the same prepared numpy image returns depth (numpy)
        returns: dict with 'equal' boolean and diff stats
        """
        pred_a = self.infer(img_input, return_numpy=True)
        # other_callable should accept HWC numpy uint8
        prepared = self._prepare_input(img_input)
        pred_b = other_callable(prepared)
        pred_b = np.asarray(pred_b, dtype=np.float32)
        same_all = np.allclose(pred_a, pred_b, atol=atol, rtol=rtol)
        diff = np.abs(pred_a - pred_b)
        return {
            'equal': bool(same_all),
            'max_abs_diff': float(diff.max()),
            'mean_abs_diff': float(diff.mean())
        }
