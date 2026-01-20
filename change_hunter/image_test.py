import os
from PIL import Image
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from change_hunter.visualization_utils import plot_results_save


def main():
    bpe_path = "/home/fhc/Project/sam3_change/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ckpt_path = "/home/fhc/Project/IDEA-Research/sam3/sam3.pt"

    image_path = "/home/fhc/Project/sam3_change/assets/images/truck.jpg"
    out_dir = "/home/fhc/Project/sam3_change/assets/test"
    save_path = os.path.join(out_dir, "0.jpg")
    os.makedirs(out_dir, exist_ok=True)

    # 1) build model
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=ckpt_path)

    # 2) load image (force RGB)
    image = Image.open(image_path).convert("RGB")

    # 3) processor + state
    processor = Sam3Processor(model, confidence_threshold=0.4)
    inference_state = processor.set_image(image)

    # 4) text prompt inference
    with torch.no_grad():
        # reset prompts (safer if it returns something)
        ret = processor.reset_all_prompts(inference_state)
        if ret is not None:
            inference_state = ret

        inference_state = processor.set_text_prompt(state=inference_state, prompt="car")

        # 如果你的 Sam3Processor 需要显式 predict，这里应加类似：
        # inference_state = processor.predict(inference_state)

    # 5) visualize/save
    plot_results_save(image, inference_state, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
