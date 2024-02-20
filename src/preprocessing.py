"""
Provides the `HousingDataset` class.

As a script, will fetch and process the images
from our raw Zillow data.
"""

import os
import json
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import requests as req
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .logging import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(CLEANED_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
META_CSV = "meta.csv"
META_PATH = os.path.join(CLEANED_DIR, META_CSV)

# Define some constants
IMAGE_SHAPE = (224, 224)
EXPECTED_IMAGES = 3045
BATCH_SIZE = 8

CHEAP = "cheap"
AVERAGE = "average"
EXPENSIVE = "expensive"

# Map from Encoding to Category and vice-versa
etoc = {
    0: CHEAP,
    1: AVERAGE,
    2: EXPENSIVE,
}
ctoe = {v: k for k, v in etoc.items()}


# Extracting the price and images
def unnest(raw_data: dict) -> dict:
    """
    Extract the main chunk of data.
    """
    return list(
        raw_data["props"]["pageProps"]["componentProps"]["gdpClientCache"].values()
    )[0]["property"]


def get_url(original_photo: dict, width: int = 1024) -> dict:
    """
    Get the link to the photo with the specified width,
    ideally a webp image.

    Raise if the expected format not found (just because
    I don't know if that ever happens and I want to find out).
    """
    assert set(original_photo.keys()) == set(
        ["caption", "mixedSources"]
    )  # just checking

    photo = original_photo["mixedSources"]

    try:
        photo = photo["webp"]  # better for compression/downloading speed
    except KeyError as e:
        raise ValueError("No webp version found") from e

    for d in photo:
        url = d["url"]
        w = d["width"]
        if int(w) == width:
            return url

    raise ValueError(f"No photo with width={width} found")


def get_urls(original_photos: list, width: int = 1024) -> list:
    """Iterate over all urls."""
    return [get_url(photo, width) for photo in original_photos]


def download_image(url: str, zpid: int, i: int) -> None:
    """
    Download the image from the specified url and save it to the specified filename.
    """
    r = req.get(url, timeout=10)
    zpid_dir = os.path.join(IMAGES_DIR, str(zpid))
    os.makedirs(zpid_dir, exist_ok=True)
    with open(os.path.join(zpid_dir, f"{str(i)}.webp"), "wb") as f:
        f.write(r.content)


def download_images(urls: list, zpid: int) -> None:
    """
    Download the images concurrently to make this a little faster.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        for i, url in enumerate(urls):
            executor.submit(download_image, url, zpid, i)


IMAGES_DIR = os.path.join(CLEANED_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def clean(verbose: bool = False) -> None:
    """
    Clean the data directory.
    NOTE this will take a pretty long time (about 2.5 hours).
    FIXME help me make this faster!
    """
    # Store zpid -> price mapping
    meta = pd.DataFrame(columns=["zpid", "price", "price_category", "num_images"])

    # For logging
    n = len(os.listdir(RAW_DIR))

    # Fetch and save property images
    downloaded = 0
    for i, fn in enumerate(os.listdir(RAW_DIR)):
        if not fn.endswith(".json"):
            continue

        with open(os.path.join(RAW_DIR, fn), "r", encoding="utf-8") as f:
            data = json.load(f)
        data = unnest(data)

        zpid = data["zpid"]
        price = data["zestimate"]
        if not price:
            if data["homeStatus"] != "FOR_SALE":
                logger.warning("Price not found for %d. Skipping.", zpid)
                continue
            price = data["price"]
            if not price:
                logger.warning("Price not found for %d. Skipping.", zpid)
                continue
        
        num_images = len(data["originalPhotos"])
        if not num_images:
            logger.warning("No images found for %d. Skipping.", zpid)
            continue

        meta.loc[len(meta)] = {  # type: ignore
            "zpid": zpid,
            "price": price,
            "price_category": None,
            "num_images": num_images,
        }

        # Skip if already downloaded
        if (
            os.path.exists(os.path.join(IMAGES_DIR, str(zpid)))
            and len(os.listdir(os.path.join(IMAGES_DIR, str(zpid)))) == num_images
        ):
            downloaded += 1
            continue

        download_images(get_urls(data["originalPhotos"]), zpid)

        if verbose and (i + 1) % 10 == 0:
            logger.info("%d/%d files processed", i + 1, n)

    if verbose:
        logger.info("Had already downloaded %d properties.", downloaded)

    # Save the zpid -> price mapping
    meta["price_category"] = pd.qcut(
        meta["price"], 3, labels=False
    )  # create our target variable
    meta.to_csv(META_PATH, index=False)

    check_corrupted_images()


def check_corrupted_images() -> None:
    """
    Some images might be corrupted and can't be opened.
    If this is the case, remove them from the dataset and
    update our metadata.

    TODO should cache somewhere which images are corrupted
    so we don't try and re-download them next time we run
    clean().
    """
    meta = pd.read_csv(META_PATH, index_col=0)
    corrupted = 0
    for zpid_str in os.listdir(IMAGES_DIR):
        zpid_dir = os.path.join(IMAGES_DIR, zpid_str)
        if not os.path.isdir(zpid_dir):
            continue
        zpid_int = int(zpid_str)
        for fn in os.listdir(zpid_dir):
            fp = os.path.join(zpid_dir, fn)
            try:
                Image.open(fp)
            except Exception as e:
                logger.warning("Corrupted image: %s", fp)
                os.remove(fp)
                corrupted += 1
                meta.loc[zpid_int, "num_images"] -= 1  # type: ignore
                # Remove the property if no images left
                if meta.loc[zpid_int, "num_images"] == 0:
                    meta.drop(zpid_int, inplace=True)
                    os.remove(zpid_dir)
    assert meta["num_images"].min() > 0
    if corrupted:
        logger.info("Removed %d corrupted images.", corrupted)
        meta.reset_index(inplace=True)
        meta.to_csv(META_PATH, index=False)


class HousingDataset(Dataset):
    """
    Implements a PyTorch Dataset for the housing data.
    """

    def __init__(
        self,
        prices_path: str,
        img_dir: str,
        transform: transforms.Compose,
        use_all_images: bool = False,
    ) -> None:
        self.meta = pd.read_csv(prices_path, index_col=0)
        self.meta = self.meta[self.meta["num_images"] > 0].reset_index()
        self.img_dir = img_dir
        self.transform = transform
        self.use_all_images = use_all_images
        self.idx_keys = self.meta["num_images"].cumsum()  # index into property

    def __len__(self) -> int:
        if self.use_all_images:
            return self.idx_keys.iloc[-1]
        return self.meta.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_all_images:
            property_index, image_cumsum = next(
                iter(self.idx_keys[self.idx_keys > idx].items())
            )
            meta = self.meta.iloc[property_index]  # type: ignore
            property_num_images = meta["num_images"]
            image_index = idx - (image_cumsum - property_num_images)
        else:
            meta = self.meta.iloc[idx]
            image_index = 0
        property_dir = os.path.join(self.img_dir, str(meta["zpid"]))
        images = os.listdir(property_dir)
        image = Image.open(os.path.join(property_dir, images[image_index]))
        tensor = self.transform(image)
        label: int = meta["price_category"]
        return tensor, label


def get_transform() -> transforms.Compose:
    """Return the transform for the dataset."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(
                lambda img: img.convert("RGB")
            ),  # in case of transparency
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO confirm the mean and std (these assume the use of a pre-trained model)
        ]
    )


def get_housing_dataset() -> HousingDataset:
    """
    Return the dataset.
    """
    return HousingDataset(
        prices_path=META_PATH,
        img_dir=IMAGES_DIR,
        transform=get_transform(),
    )


def main() -> None:
    """
    Download all the images and preprocess a sample batch
    of the dataset.
    """
    clean(verbose=True)
    logger.info("Data preprocessing complete.")
    housing_dataset = HousingDataset(
        prices_path=META_PATH,
        img_dir=IMAGES_DIR,
        transform=get_transform(),
    )
    data_loader = DataLoader(
        housing_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )  # FIXME why can't I raise num_workers?
    _image, category = next(iter(data_loader))
    _image = _image[0]
    category = category[0].item()
    logger.info("Sample Data")
    logger.info("------------")
    logger.info("Price encoding: %d", category)
    logger.info("Price category: %s", etoc[category])
    logger.info("Image shape: %s", _image.shape)
    logger.info("Image[0] (224, 224) tensor for R:\n%s", _image[0])


if __name__ == "__main__":
    main()
