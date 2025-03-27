import os
import numpy as np
import torch as th
import torchvision.utils as vtils
from PIL import Image as PILImage

from utils.noise import get_noise
from utils.tools import check_dims, cycle
from utils.svd_operators import get_degradation

from accelerate import Accelerator
from ema_pytorch import EMA

"""
这段Python代码定义了一个名为EvalLoop的类
主要用于评估一个隐式模型（implicit model）和扩散模型（diffusion model）在图像重建任务中的性能。
"""


class EvalLoop:
    def __init__(
            self,
            *,
            implicit_model,  # 隐式模型，用于图像重建。
            diffusion,  # 扩散模型，用于生成或去噪图像。
            batch_size,
            eval_data,  # 评估数据集，包含干净的图像。
            y_data,  # 测量数据（模糊或降质图像）。如果为None，则通过get_measurements生成。
            perturb_h,  # 扰动参数，用于扩散过程。
            ckpt,  # 检查点
            save_dir,  # 保存目录
            deg,  # 降质类型（如去噪、超分辨率、上色等）。
    ):
        self.implicit_model = implicit_model
        self.diffusion = diffusion

        self.batch_size = batch_size
        self.eval_data = eval_data
        if y_data is not None:
            self.y_data = cycle(y_data)
        else:
            self.y_data = y_data

        self.perturb_h = perturb_h
        self.save_dir = save_dir
        self.deg = deg

        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision='no'
        )
        self.accelerator.native_amp = False  # 初始化Accelerator（用于混合精度训练和分布式训练，但此处禁用混合精度）。
        self.device = self.accelerator.device
        self.ema = EMA(self.implicit_model)  # 加载EMA（指数移动平均）模型权重。
        self.load(ckpt)

        # 根据降质类型（deg）初始化噪声生成器（noiser）和降质算子（operator）。
        if self.deg == 'deno':
            self.noiser = get_noise(name='gaussian', sigma=0.20)
        else:
            self.noiser = get_noise(name='gaussian', sigma=0.05)
        self.operator = get_degradation(self.deg, self.device)

    """
    遍历评估数据集，对每张图像进行以下操作：
    获取测量数据（模糊或降质图像）。
    使用扩散模型和隐式模型重建图像。
    保存测量图像和重建图像。
    """

    def run_loop(self):
        img_cnt = 0
        for idx, (clean, _) in enumerate(self.eval_data):
            clean = clean * 2 - 1  # 将图像从[0,1]归一化到[-1,1]

            if self.y_data is None:  # 如果y_data为None，调用get_measurements生成模糊图像。否则直接从y_data中读取。
                blur = self.get_measurements(clean.to(self.device))
            else:
                blur = next(self.y_data)
                blur = blur[:, 0].to(self.device)

            blur2 = check_dims(clean, blur, self.operator)  # 使用扩散模型对模糊图像加噪（模拟反向扩散过程）。
            t = th.ones((blur2.shape[0],), dtype=th.int, device=self.device) * 999
            blur2 = self.diffusion.q_sample_i2sb(t, blur2, blur2, self.perturb_h)

            # 通过扩散模型的均值方差预测和去噪步骤生成重建图像。
            with self.accelerator.autocast():
                with th.no_grad():
                    img = self.diffusion.p_mean_variance(self.ema.ema_model, blur2, t, clip_denoised=False)[
                        'model_output']
                    img = self.diffusion.compute_pred_x0(t, blur2, img, False)
                    img = th.nn.functional.tanh(img)  # # 将输出限制到[-1,1]

            for i in range(clean.shape[0]):
                vtils.save_image(blur[i], os.path.join(self.save_dir, 'measurement', f'{img_cnt}.png'), normalize=True)
                recon = (img[i] + 1.) / 2.
                image_np = recon.data.cpu().numpy().transpose(1, 2, 0)
                image_np = PILImage.fromarray((image_np * 255).astype(np.uint8))
                image_np.save(os.path.join(self.save_dir, 'recon', f'{img_cnt}.png'))
                img_cnt += 1
            # 保存测量图像（blur）和重建图像（recon）到指定目录。
            print(f"{img_cnt} th image generated...")

    # 从检查点加载EMA模型的权重。
    def load(self, ckpt):
        checkpoint = th.load(ckpt, map_location='cpu')  # ckpt: 检查点文件路径。
        self.ema.load_state_dict(checkpoint['implicit_func_ema'])
        self.ema.to(self.device)
        self.ema.ema_model.eval()  # 将模型设置为评估模式（禁用dropout等训练专用层）。

    # 根据降质类型（deg）对干净图像（clean）进行降质处理，并添加噪声。
    def get_measurements(self, clean):
        b, c, h, w = clean.shape

        if self.deg == 'sr_averagepooling' or self.deg == 'bicubic':
            h, w = int(h / 4), int(w / 4)
        elif self.deg == 'colorization':
            c = 1

        if self.deg == 'inpainting':
            blur = self.operator.A(clean).clone().detach()
            blur_pinv = self.operator.At(blur).view(b, c, h, w)
            blur = blur_pinv
        else:
            blur = self.operator.A(clean).view(b, c, h, w).clone().detach()
        blur = self.noiser(blur)  # Additive noise
        blur = blur.to(self.device)

        return blur
