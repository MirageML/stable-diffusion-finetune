
from ldm.simplet2i import T2I
import argparse
import shutil
import subprocess
import os
from PIL import Image
import glob

"""
T2I(weights     = <path>        // models/ldm/stable-diffusion-v1/model.ckpt
    config      = <path>        // configs/stable-diffusion/v1-inference.yaml
    iterations  = <integer>     // how many times to run the sampling (1)
    steps       = <integer>     // 50
    seed        = <integer>     // current system time
    sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
    grid        = <boolean>     // false
    width       = <integer>     // image width, multiple of 64 (512)
    height      = <integer>     // image height, multiple of 64 (512)
    cfg_scale   = <float>       // unconditional guidance scale (7.5)
)
"""

def finetune(init_word, image_data, style, output_path):
    size = (512, 512)

    # get all the files in a folder, make sure all are image files
    files = glob.glob(f'{image_data}/*')
    os.makedirs("./training_data", exist_ok=True)

    for fil in files:
        # implement file type checking here if required
        if not fil.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        # get the basename, e.g. "dragon.jpg" -> ("dragon", ".jpg")
        basename = os.path.splitext(os.path.basename(fil))[0]

        with Image.open(fil) as img:
            # resize the image to 512 x 512
            img = img.resize(size)
            # rotate the image if required
            # img = img.rotate(90)

            # save the resized image, modify the resample method if required, modify the output directory as well
            img.save(f"./training_data/{basename}.png", format="PNG", resample=Image.Resampling.NEAREST)


    base = "./configs/stable-diffusion/v1-finetune.yaml"
    if style:
        base = "./configs/stable-diffusion/v1-finetune_style.yaml"
    command = f"python main.py --base {base} --train true --actual_resume ./models/ldm/stable-diffusion-v1/model.ckpt -n {init_word} --gpus 0, --train_steps 7000 --data_root ./training_data/ --init_word {init_word}"

    subprocess.call(command.split(" "))

    embedding_file = "embeddings.pt"
    embedding_path = ""
    for root, dirs, files in os.walk("logs"):
        for name in files:
            if name == embedding_file:
                embedding_path = os.path.abspath(os.path.join(root, name))
                break

    shutil.copy(embedding_path, os.path.join(output_path, "embeddings.pt"))

def create_finetuned_data(init_word, prompt, num_images, init_image, output_path):
    # model configuration
    t2i = T2I(
        weights='./models/ldm/stable-diffusion-v1/model.ckpt',
        config='./configs/stable-diffusion/v1-inference.yaml',
        embedding_path=os.path.join(output_path, "embeddings.pt")  # modify the embedding path
        seed=None,              # seed for random number generator
        cfg_scale=7.5,          # how strongly the prompt influences the image (7.5) (must be >1)
        width=512,              # width of image, in multiples of 64 (512)
        height=512,             # height of image, in multiples of 64 (512)
        sampler_name='ddim',    # ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
        ddim_eta=0.0,           # deterministic image randomness (eta=0.0 means the same seed always produces the same image)
        strength=0.75,          # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
    )

    seed = t2i.seed

    # model loading
    t2i.load_model()

    # variable initialization
    if init_word not in prompt:
        raise ValueError(f"{init_word} must be in the prompt")

    # inference returns a list of tuple
    results = t2i.prompt2image(
        prompt    = prompt,
        outdir    = output_path,
        iterations=num_images,
        steps     = 50,
        init_img  = init_image,
        # gfpgan_strength                 // strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
    )

    # save the image in outputs folder
    for row in results:
        im   = row[0]
        seed = row[1]
        im.save(os.path.join(output_path, f'image-{seed}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune Images")
    parser.add_argument("--init_word", type=str, required=True, help="Finetune word")
    parser.add_argument("--dataset_path", type=str, help="Path to Dataset")
    parser.add_argument("--style", type=bool, default=False, help="Style?")
    parser.add_argument("--output_path", type=str, default="output", help="Path to output directory")
    parser.add_argument("--train_model", type=bool, default=False, help="Flag to train_model or generate data")

    parser.add_argument("--prompt", type=str, help="prompt")
    parser.add_argument("--init_image", type=str, default=None, help="init_img")
    parser.add_argument("--num_images", type=int, default=10, help="num_images")

    args, unknown_args = parser.parse_known_args()

    output_path = os.path.abspath(args.output_path)

    os.makedirs(output_path, exist_ok=True)
    if args.init_image:
        args.init_image = os.path.abspath(args.init_img)

    # args.init_word = "airfield"
    # args.dataset_path = os.path.abspath("training_data/key")
    if args.train_model:
        finetune(args.init_word, os.path.abspath(args.dataset_path), args.style, output_path)
    else:
        create_finetuned_data(args.init_word, args.prompt, args.num_images, args.init_image, output_path)
