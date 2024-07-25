import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
import json
from PIL import Image
import random
import numpy as np
import os


class MyDataset(Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        size=512,
        image_root_path="",
    ):
        super().__init__
        self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))

        self.target_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        item = self.data[index]  # single object from json file
        text = item["item"]
        target_img = item["target"]
        source_1 = item["source_1"]
        source_2 = item["source_2"]

        raw_target_img = Image.open(os.path.join(self.image_root_path, target_img)).convert("RGB")
        raw_source_1_img = Image.open(os.path.join(self.image_root_path, source_1)).convert("RGB")
        raw_source_2_img = Image.open(os.path.join(self.image_root_path, source_2)).convert("RGB")

        target_img_tensor = self.target_transforms(raw_target_img)
        source_1_img_tensor = self.conditioning_transforms(raw_source_1_img)
        source_2_img_tensor = self.conditioning_transforms(raw_source_2_img)

        prompt = f"a photo of {text}"

        # get text and tokenize
        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # drop
        # drop_image_embed = 0
        # rand_num = random.random()
        # if rand_num < self.i_drop_rate:
        #     drop_image_embed = 1
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
        #     text = ""
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
        #     text = ""
        #     drop_image_embed = 1

        return {
             "target_img":target_img_tensor,
             "controlnet_1_img": source_1_img_tensor,
             "controlnet_2_img":source_2_img_tensor,
             "text_input_ids": text_input_ids
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    target_imgs = torch.stack([example["target_img"] for example in data])
    controlnet_1_imgs = torch.stack([example["controlnet_1_img"] for example in data])
    controlnet_2_imgs = torch.stack([example["controlnet_2_img"] for example in data])
    text_input_ids = torch.cat(
        [example["text_input_ids"] for example in data], dim=0
    )

    return {
        "target_imgs":target_imgs,
        "controlnet_1_imgs":controlnet_1_imgs,
        "controlnet_2_imgs":controlnet_2_imgs,
        "text_input_ids":text_input_ids
    }


if __name__ == "__main__":

    from transformers import CLIPTokenizer

    pretrained_model_name = "runwayml/stable-diffusion-v1-5"
    json_file = "/workspace/control_net_data/control_net_train.json"
    image_root_path = "/workspace/control_net_data"

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer"
    )

    dataset = MyDataset(
        json_file=json_file,
        tokenizer=tokenizer,
        image_root_path=image_root_path,
    )
    # shapes after using dataloader with collate function
    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=collate_fn, batch_size=4, num_workers=6
    )
    print("\n------using data loader with collate function--------")
    print(f"Lenght of dataloader with batchsize 4: {len(train_dataloader)}")
    for batch in train_dataloader:
        print(f"Shape of target image: {batch['target_imgs'].shape}")
        print(f"Shape of controlnet source 1: {batch['controlnet_1_imgs'].shape}")
        print(f"Shape of controlnet source 2: {batch['controlnet_2_imgs'].shape}")
        print(f"Shape of text tokens: {batch['text_input_ids'].shape}")
        break