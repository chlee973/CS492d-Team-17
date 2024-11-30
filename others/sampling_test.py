import numpy as np
import torch
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image
from sketch_diffusion.model import DiffusionModule
from sketch_diffusion.scheduler import DDPMScheduler
from pathlib import Path
from torchvision.utils import save_image
import argparse

# main을 함수화하여 재사용 가능하도록 변경
def run_test_sampling(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    images_dir = save_dir
    images_dir.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
    ).to(device)

    if args.sample_method == 'ddpm':
        num_inference_timesteps = num_train_timesteps
        eta = 1.0
    else:
        num_inference_timesteps = args.num_inference_timesteps
        eta = 0.0

    total_num_samples = args.total_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    sample_count = 0
    for i in range(num_batches):
        B = min(args.batch_size, total_num_samples - sample_count)

        if args.use_cfg:
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            vectors = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=args.cfg_scale,
            )
        else:
            vectors = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                num_inference_timesteps=num_inference_timesteps,
                eta=eta,
                guidance_scale=0.0,
            )
        pen_states = torch.ones((vectors.shape[0], vectors.shape[1], 1), device=vectors.device)
        samples = torch.cat((vectors, pen_states), dim=-1)
        samples = samples.cpu()
        
        for sample in samples:
            sketch_pil = tensor_to_pil_image(sample, canvas_size=(args.img_size,)*2)
            image_path = images_dir / f"sample_{sample_count:06d}.png"
            sketch_pil.save(str(image_path))
            sample_count += 1

    print(f"{sample_count}개의 이미지가 '{images_dir}' 폴더에 저장되었습니다.")


def main():
    # 필요한 변수들을 설정합니다.
    save_dir = Path("results/diffusion-ddim-11-28-182512-cat_1000step_vectors_transformer_16head")  # 실제 저장 디렉토리 경로로 변경하세요.
    num_inference_timesteps = 1000  # 원하는 타임스텝 수로 설정하세요.

    # args_test 네임스페이스를 생성합니다.
    args_test = argparse.Namespace(
        ckpt_path=str(save_dir / "last.ckpt"),
        save_dir=str(save_dir / f"step=-test"),
        sample_method="ddim",
        gpu=0,
        batch_size=4,
        total_samples=8,
        img_size=256,
        use_cfg=False,
        num_inference_timesteps=100,
        cfg_scale=7.5,
        beta_1=1e-5,
        beta_T=0.01,
    )

    # 샘플링 함수를 호출합니다.
    run_test_sampling(args_test)

if __name__ == "__main__":
    main()