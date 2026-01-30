import argparse
import torch
import os
import sys
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import logging
import time


try:
    from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from mplug_owl2.conversation import conv_templates, SeparatorStyle
    from mplug_owl2.model.builder import load_pretrained_model
    from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
        KeywordsStoppingCriteria
except ImportError as e:
    print(f"Error importing from mplug_owl2: {e}")
    print("Please ensure mplug_owl2 is in your PYTHONPATH or installed correctly.")
    print("You might need to add its path like in the commented out section above.")
    sys.exit(1)

LOG_FORMAT = '%(asctime)s [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

VIEWS_BATCH_SIZE = 8



def disable_torch_init():
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    logger.info("Disabled redundant torch default initialization.")


def load_image_pil(image_file_path):
    try:
        image = Image.open(image_file_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_file_path}: {e}")
        return None


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_view_filenames_from_dir(scene_dir_path):

    view_files = []
    if not os.path.isdir(scene_dir_path):
        logger.warning(f"Scene directory not found: {scene_dir_path}")
        return view_files
    for fname in sorted(os.listdir(scene_dir_path)):
        if fname.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg')):
            view_files.append(fname)
    if len(view_files) != 81:
        logger.warning(f"Expected 81 views in {scene_dir_path}, but found {len(view_files)}.")
    return view_files


def extract_features_for_dataset(dataset_key, dataset_config, global_config, tokenizer, model, image_processor,
                                 prompt_text_template, device_mplug):
    logger.info(f"--- Starting feature extraction for dataset: {dataset_key} ---")

    base_path = dataset_config.base_path
    dist_dir_rel = dataset_config.dist_dir
    full_dist_path = os.path.join(base_path, dist_dir_rel)


    feature_save_parent_dir = global_config.lmm_feature_dir

    dataset_feature_subdir = dataset_config.feature_subdir
    target_feature_save_dir = os.path.join(feature_save_parent_dir, dataset_feature_subdir)
    os.makedirs(target_feature_save_dir, exist_ok=True)
    logger.info(f"Features will be saved to: {target_feature_save_dir}")

    if not os.path.isdir(full_dist_path):
        logger.error(f"Distorted images path not found for {dataset_key}: {full_dist_path}")
        return

    scene_names = sorted([d for d in os.listdir(full_dist_path) if os.path.isdir(os.path.join(full_dist_path, d))])
    if not scene_names:
        logger.warning(f"No scene subdirectories found in {full_dist_path}")
        return

    logger.info(f"Found {len(scene_names)} scenes in {dataset_key}.")


    conv_mode = global_config.get("mplug_conv_mode", "mplug_owl2")
    conv = conv_templates[conv_mode].copy()
    inp = prompt_text_template + "\n" + DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_tokens = conv.get_prompt()

    input_ids_template = tokenizer_image_token(prompt_tokens, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(
        device_mplug)

    for scene_name in tqdm(scene_names, desc=f"Processing Scenes in {dataset_key}"):
        current_scene_dist_path = os.path.join(full_dist_path, scene_name)

        target_feature_file_path = os.path.join(target_feature_save_dir, f"scene_{scene_name}_features.pt")
        if os.path.exists(target_feature_file_path) and not global_config.get("overwrite_features", False):
            logger.debug(f"Features for {scene_name} already exist. Skipping.")
            continue

        view_filenames = get_view_filenames_from_dir(current_scene_dist_path)
        if len(view_filenames) == 0:
            logger.warning(f"No view images found in {current_scene_dist_path} for scene {scene_name}. Skipping.")
            continue
        if len(view_filenames) != 81:
            logger.warning(
                f"Scene {scene_name} in {dataset_key} has {len(view_filenames)} views, expected 81. Processing available views.")

        all_view_feature_tensors = []

        for i in range(0, len(view_filenames), VIEWS_BATCH_SIZE):
            batch_view_filenames = view_filenames[i: i + VIEWS_BATCH_SIZE]
            batch_pil_images = []
            valid_view_in_batch_count = 0

            for view_fname in batch_view_filenames:
                view_fpath = os.path.join(current_scene_dist_path, view_fname)
                pil_img = load_image_pil(view_fpath)
                if pil_img:
                    pil_img_sq = expand2square(pil_img, tuple(int(x * 255) for x in image_processor.image_mean))
                    batch_pil_images.append(pil_img_sq)
                    valid_view_in_batch_count += 1
                else:
                    logger.warning(f"Skipping view {view_fname} due to loading error.")

            if not batch_pil_images:
                continue
            image_tensors_processed = image_processor.preprocess(batch_pil_images, return_tensors='pt')[
                'pixel_values'].half().to(device_mplug)
            current_batch_input_ids = input_ids_template.repeat(valid_view_in_batch_count, 1)

            try:
                with torch.inference_mode():

                    output_hidden_states = model(
                        current_batch_input_ids,
                        images=image_tensors_processed,
                        output_hidden_states=True
                    )['hidden_states'][-1]

                    output_view_features = torch.mean(output_hidden_states, dim=1, keepdim=True)
                    all_view_feature_tensors.extend(list(torch.unbind(output_view_features.cpu(), dim=0)))

            except Exception as e:
                logger.error(f"Error during model inference for a batch in scene {scene_name}: {e}")
                continue

        if len(all_view_feature_tensors) > 0:
            try:
                scene_features_tensor = torch.stack(all_view_feature_tensors).squeeze(1)
                if scene_features_tensor.shape[0] < 81:
                    logger.warning(
                        f"Scene {scene_name} resulted in {scene_features_tensor.shape[0]}/81 features. Padding with zeros.")
                    padding_needed = 81 - scene_features_tensor.shape[0]
                    padding_tensor = torch.zeros((padding_needed, scene_features_tensor.shape[1]),
                                                 dtype=scene_features_tensor.dtype)
                    scene_features_tensor = torch.cat([scene_features_tensor, padding_tensor], dim=0)
                elif scene_features_tensor.shape[
                    0] > 81:
                    logger.warning(
                        f"Scene {scene_name} resulted in {scene_features_tensor.shape[0]} > 81 features. Truncating.")
                    scene_features_tensor = scene_features_tensor[:81, :]

                torch.save(scene_features_tensor, target_feature_file_path)
                logger.debug(
                    f"Saved features for scene {scene_name} to {target_feature_file_path} with shape {scene_features_tensor.shape}")
            except Exception as e:
                logger.error(f"Error stacking or saving features for scene {scene_name}: {e}")
        else:
            logger.warning(f"No view features extracted for scene {scene_name}. No .pt file saved.")

    logger.info(f"--- Finished feature extraction for dataset: {dataset_key} ---")


def main():
    parser = argparse.ArgumentParser(description="LMM Feature Extraction for Light Fields")
    parser.add_argument('--config', type=str, default='configs/combined.yaml',
                        help="Path to the combined configuration file.")
    parser.add_argument('--device', type=str, default=None, help="Override device (e.g., 'cuda:0', 'cpu')")
    parser.add_argument('--overwrite_features', action='store_true', help="Overwrite existing feature files.")
    cli_args = parser.parse_args()

    try:
        cfg_combined = OmegaConf.load(cli_args.config)
        logger.info(f"Loaded combined configuration from: {cli_args.config}")
    except Exception as e:
        logger.error(f"Error loading combined config file {cli_args.config}: {e}")
        return

    device_mplug_str = cli_args.device if cli_args.device else cfg_combined.get("mplug_device", "cuda")
    if "cuda" in device_mplug_str and not torch.cuda.is_available():
        logger.warning("CUDA specified but not available. Falling back to CPU.")
        device_mplug_str = "cpu"
    device_mplug = torch.device(device_mplug_str)
    logger.info(f"Using device for mPLUG-Owl2: {device_mplug}")

    if cli_args.overwrite_features:
        cfg_combined.overwrite_features = True

    disable_torch_init()

    model_path = cfg_combined.mplug_model_path  # e.g., "MAGAer13/mplug-owl2-llama2-7b"
    model_base = cfg_combined.get("mplug_model_base", None)
    load_8bit = cfg_combined.get("mplug_load_8bit", False)
    load_4bit = cfg_combined.get("mplug_load_4bit", False)

    logger.info(f"Loading mPLUG-Owl2 model: {model_path} (Base: {model_base})")
    try:
        model_name_from_path = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name_from_path, load_8bit, load_4bit, device=device_mplug
        )
        model.eval()  # Set model to evaluation mode
        logger.info("mPLUG-Owl2 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading mPLUG-Owl2 model: {e}")
        return

    prompt_lf_quality = "Analyze this image, which is a single view from a light field capture. Assess its overall visual quality."
    logger.info(f"Using prompt: \"{prompt_lf_quality}\"")

    start_time_total = time.time()
    # for dset_key in cfg_combined.all_dataset_keys:  # e.g., "NBU", "SHU", "Win5LID"
    for dset_key in cfg_combined.all_dataset_keys_for_feature_extraction:
        if dset_key in cfg_combined.datasets:
            dataset_specific_config = cfg_combined.datasets[dset_key]

            # --- MODIFIED: Simplified Win5LID handling ---
            if dset_key == "Win5LID":
                # Win5LID is now treated as a single entity for feature extraction
                # dataset_specific_config here is cfg_combined.datasets.Win5LID
                # The feature_subdir should be "Win5LID"
                extract_features_for_dataset(
                    dataset_key=dset_key,  # Use "Win5LID" for subdir name
                    dataset_config=dataset_specific_config,
                    global_config=cfg_combined,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    prompt_text_template=prompt_lf_quality,
                    device_mplug=device_mplug
                )
            else:
                extract_features_for_dataset(
                    dataset_key=dset_key,
                    dataset_config=dataset_specific_config,
                    global_config=cfg_combined,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    prompt_text_template=prompt_lf_quality,
                    device_mplug=device_mplug
                )
        else:
            logger.warning(f"Configuration for dataset key '{dset_key}' not found in combined.yaml. Skipping.")

    end_time_total = time.time()
    logger.info(f"Total feature extraction time: {end_time_total - start_time_total:.2f} seconds.")
    logger.info("All feature extraction processes finished.")


if __name__ == "__main__":
    main()