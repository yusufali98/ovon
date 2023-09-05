import argparse
import glob
import os

import clip
import numpy as np
import torch
from tqdm import tqdm

from ovon.utils.utils import load_dataset, save_pickle

PROMPT = "{category}"


def tokenize_and_batch(clip, goal_categories):
    tokens = []
    for category in goal_categories:
        prompt = PROMPT.format(category=category)
        tokens.append(clip.tokenize(prompt, context_length=77).numpy())
    return torch.tensor(np.array(tokens)).cuda()


def save_to_disk(text_embedding, goal_categories, output_path):
    output = {}
    for goal_category, embedding in zip(goal_categories, text_embedding):
        output[goal_category] = embedding.detach().cpu().numpy()
    save_pickle(output, output_path)


def cache_embeddings(goal_categories, output_path, clip_model="RN50"):
    model, _ = clip.load(clip_model)
    batch = tokenize_and_batch(clip, goal_categories)

    with torch.no_grad():
        print(batch.shape)
        text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    save_to_disk(text_embedding, goal_categories, output_path)


def load_categories_from_dataset(path):
    files = glob.glob(os.path.join(path, "*json.gz"))

    categories = []
    for f in tqdm(files):
        dataset = load_dataset(f)
        for goal_key in dataset["goals_by_category"].keys():
            categories.append(goal_key.split("_")[1])
    return list(set(categories))


def main():
# def main(dataset_path, output_path):

    dataset_path = "/srv/kira-lab/share4/yali30/cow_ovon/hm3d_data/datasets/ovon_new/v3/train/content"

    goal_categories = load_categories_from_dataset(dataset_path)
    
    # val_seen_categories = load_categories_from_dataset("/srv/kira-lab/share4/yali30/cow_ovon/hm3d_data/datasets/ovon_naoki/ovon/hm3d/v3_shuffled_cleaned/val_seen/content")
    # val_unseen_easy_categories = load_categories_from_dataset("/srv/kira-lab/share4/yali30/cow_ovon/hm3d_data/datasets/ovon_naoki/ovon/hm3d/v3_shuffled_cleaned/val_unseen_easy/content")
    # val_unseen_hard_categories = load_categories_from_dataset("/srv/kira-lab/share4/yali30/cow_ovon/hm3d_data/datasets/ovon_naoki/ovon/hm3d/v3_shuffled_cleaned/val_unseen_hard/content")
    
    val_seen_categories = []
    val_unseen_easy_categories = []
    val_unseen_hard_categories = []

    # goal_categories.extend(val_seen_categories)
    # goal_categories.extend(val_unseen_easy_categories)
    # goal_categories.extend(val_unseen_hard_categories)

    print("Total goal categories: {}".format(len(goal_categories)))
    print(
        "Train categories: {}, Val seen categories: {}, Val unseen easy categories: {}, Val unseen hard categories: {}".format(
            len(goal_categories),
            len(val_seen_categories),
            len(val_unseen_easy_categories),
            len(val_unseen_hard_categories),
        )
    )

    output_path = "/srv/kira-lab/share4/yali30/ovon_duplicate/ovon/ovon_stretch_cache_ram_only_train.pkl"
    cache_embeddings(goal_categories, output_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset-path",
    #     type=str,
    #     required=True,
    #     help="file path of OVON dataset",
    # )
    # parser.add_argument(
    #     "--output-path",
    #     type=str,
    #     required=True,
    #     help="output path of clip features",
    # )
    # args = parser.parse_args()
    # main(args.dataset_path, args.output_path)
    main()
