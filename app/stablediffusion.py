from this import s
from types import SimpleNamespace
import os
import cv2
import time
import datetime
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

class StableDiffusion:
    def __init__(self, stable_diffusion_path: str) -> None:
        print('Stable Diffusion init...', stable_diffusion_path)
        self.stable_diffusion_path = stable_diffusion_path
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        model.cuda()
        model.eval()
        return model

    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img

    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def text2img(self, text: str, uid: int, username: str, count: int = 1) -> None:
        ts = int(time.time())
        date_time = datetime.datetime.fromtimestamp(ts)
        dt_string = date_time.strftime("%D/%m/%Y %H:%M:%S")

        opt = SimpleNamespace(**{
            # 'prompt': 'a photo of a ' + text, # the prompt to render
            # 'prompt': 'a painting of a ' + text, # the prompt to render
            'prompt': text,          # the prompt to render
            'outdir': f'output/{uid}', # dir to write results to
            'ddim_steps': 50,        # number of ddim sampling steps
            'plms': True,            # use plms sampling
            'laion400m': False,      # uses the LAION400M model
            'fixed_code': False,     # if enabled, uses the same starting code across samples
            'ddim_eta': 0.0,         # ddim eta (eta=0.0 corresponds to deterministic sampling
            'n_iter': count,             # sample this often
            'H': 512,                # image height, in pixel space
            'W': 512,                # image width, in pixel space
            'C': 4,                  # latent channels
            'f': 8,                  # downsampling factor
            'n_samples': 1,          # how many samples to produce for each given prompt. A.k.a. batch size",
            'n_rows': 0,             # rows in the grid (default: n_samples)"
            'scale': 7.5,            # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
            'from_file': '',         # if specified, load prompts from this file
            'config': os.path.join(self.stable_diffusion_path, 'configs/stable-diffusion/v1-inference.yaml'), # path to config which constructs model
            'ckpt': os.path.join(self.stable_diffusion_path, 'models/ldm/stable-diffusion-v1/model.ckpt'), # path to checkpoint of model
            'seed': 42,              # the seed (for reproducible sampling)
            'precision': 'autocast', # evaluate at this precision
        })

        print('Options: ', opt)

        if opt.laion400m:
            print("Falling back to LAION 400M model...")
            opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            opt.ckpt = "models/ldm/text2img-large/model.ckpt"
            opt.outdir = "outputs/txt2img-samples-laion400m"

        # seed_everything(opt.seed)

        config = OmegaConf.load(f"{opt.config}")
        model = self.load_model_from_config(config, f"{opt.ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.chunk(data, batch_size))

        images_path = os.path.join(outpath, "images")
        os.makedirs(images_path, exist_ok=True)

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        filepaths = []

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = self.put_watermark(img, wm_encoder)
                                image_filename = f'{ts}.png'
                                image_filepath = os.path.join(images_path, image_filename)
                                img.save(image_filepath)
                                filepaths.append(image_filepath)
                                query_filename = f'queries.md'
                                query_filepath = os.path.join(outpath, query_filename)
                                query_file_exists = os.path.isfile(query_filepath)
                                with open(query_filepath, "a+") as query_file:
                                    if not query_file_exists:
                                        query_file.writelines([
                                            f'# [{username}](https://t.me/{username}) ({uid})\n\n',
                                        ])
                                    query_file.writelines([
                                        f'## {text}\n\n',
                                        f'![text](images/{ts}.png)\n\n',
                                        f'{dt_string}\n\n',
                                        '---\n\n',
                                    ])

                            all_samples.append(x_checked_image_torch)

                    toc = time.time()

        return filepaths
