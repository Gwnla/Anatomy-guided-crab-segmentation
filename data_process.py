import os, json, random, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CLASS_NAMES = ['body', 'left_claw', 'right_claw', 'left_legs', 'right_legs']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ========== Data Augmentation ==========
class WeakAugmentation:
    def __call__(self, image, mask=None):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            if mask is not None:
                cls_order = [0, 2, 1, 4, 3]
                mask = mask[cls_order]
                mask = np.ascontiguousarray(np.flip(mask, axis=2))
        h, w = image.shape[:2]
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        if mask is not None:
            mask_rot = [cv2.warpAffine(mask[c], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for c in range(mask.shape[0])]
            mask = np.stack(mask_rot, axis=0)
        scale = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask_resized = [cv2.resize(mask[c].astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST) for c in range(mask.shape[0])]
            mask = np.stack(mask_resized, axis=0)
        if new_h < h or new_w < w:
            pad_top, pad_left = (h - new_h)//2, (w - new_w)//2
            image = cv2.copyMakeBorder(image, pad_top, h-new_h-pad_top, pad_left, w-new_w-pad_left, cv2.BORDER_CONSTANT, value=0)
            if mask is not None:
                mask_padded = np.zeros((mask.shape[0], h, w), dtype=mask.dtype)
                for c in range(mask.shape[0]):
                    mask_padded[c] = cv2.copyMakeBorder(mask[c], pad_top, h-new_h-pad_top, pad_left, w-new_w-pad_left, cv2.BORDER_CONSTANT, value=0)
                mask = mask_padded
        elif new_h > h or new_w > w:
            crop_top, crop_left = (new_h-h)//2, (new_w-w)//2
            image = image[crop_top:crop_top+h, crop_left:crop_left+w]
            if mask is not None:
                mask = mask[:, crop_top:crop_top+h, crop_left:crop_left+w]
        alpha = 1 + random.uniform(-0.1, 0.1)
        beta = random.uniform(-0.1, 0.1) * 255
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if mask is not None:
            return image, mask
        return image

class StrongAugmentation:
    def __call__(self, image):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if new_h < h or new_w < w:
            pad_top, pad_left = (h-new_h)//2, (w-new_w)//2
            image = cv2.copyMakeBorder(image, pad_top, h-new_h-pad_top, pad_left, w-new_w-pad_left, cv2.BORDER_CONSTANT, value=0)
        elif new_h > h or new_w > w:
            crop_top, crop_left = (new_h-h)//2, (new_w-w)//2
            image = image[crop_top:crop_top+h, crop_left:crop_left+w]
        if random.random() < 0.8:
            image = cv2.convertScaleAbs(image, alpha=1+random.uniform(-0.2, 0.2), beta=random.uniform(-0.2, 0.2)*255)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-30, 30)) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.8, 1.2), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        if random.random() < 0.3:
            image = np.clip(image.astype(np.float32) + np.random.randn(*image.shape) * 20, 0, 255).astype(np.uint8)
        return image

# ========== Dataset ==========
def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std

class LabeledDataset(Dataset):
    def __init__(self, data_dir, image_size=1024, augment=True):
        self.image_size = image_size
        self.augment = augment
        self.augmentor = WeakAugmentation() if augment else None
        meta_path = os.path.join(data_dir, 'metadata.json')
        self.metadata = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = cv2.imread(item['image_path'])
        if image is None: image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(item['mask_path']).astype(np.float32)
        if self.augment and self.augmentor is not None:
            h, w = image.shape[:2]
            leg_mask = np.zeros((h, w), dtype=bool)
            for idx in [3, 4]: leg_mask |= mask[idx] > 0
            if leg_mask.any() and random.random() < 0.5:
                ys, xs = np.where(leg_mask)
                y_min, y_max, x_min, x_max = max(0, ys.min()-int((ys.max()-ys.min())*0.3)), min(h, ys.max()+int((ys.max()-ys.min())*0.3)), max(0, xs.min()-int((xs.max()-xs.min())*0.3)), min(w, xs.max()+int((xs.max()-xs.min())*0.3))
                image = cv2.resize(image[y_min:y_max, x_min:x_max], (w, h), interpolation=cv2.INTER_LINEAR)
                mask_resized = [cv2.resize(mask[c, y_min:y_max, x_min:x_max].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) for c in range(mask.shape[0])]
                mask = np.stack(mask_resized, axis=0)
            image, mask = self.augmentor(image, mask)
        image = normalize_image(image)
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(mask).float()

class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, image_size=1024, strong_aug=True):
        self.image_size = image_size
        self.geo_aug = GeometricAugmentation()
        self.weak_photo = WeakPhotometricAugmentation()
        self.strong_photo = StrongPhotometricAugmentation() if strong_aug else None
        self.image_paths = []
        filelist_path = os.path.join(data_dir, 'filelist.txt')
        if os.path.exists(filelist_path):
            with open(filelist_path, 'r') as f:
                self.image_paths = [line.strip() for line in f if line.strip()]
        if not self.image_paths:
            for fname in sorted(os.listdir(data_dir)):
                if fname.lower().endswith(('.jpg','.jpeg','.png')): self.image_paths.append(os.path.join(data_dir, fname))
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None: image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        geo_image = self.geo_aug(image.copy())
        image_w = self.weak_photo(geo_image.copy())
        image_s = self.strong_photo(geo_image.copy()) if self.strong_photo else geo_image.copy()
        return normalize_image(image_w).transpose(2,0,1).copy(), normalize_image(image_s).transpose(2,0,1).copy(), self.image_paths[idx]

# ========== Preprocessing ==========
def parse_x_anylabeling_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_path = os.path.join(os.path.dirname(json_path), data.get('imagePath', ''))
    polygons = {}
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        if label in CLASS_TO_IDX and len(shape.get('points', [])) >= 3:
            polygons[label] = np.array(shape['points'], dtype=np.int32)
    return image_path, polygons

def polygons_to_mask(polygons, height, width):
    mask = np.zeros((len(CLASS_NAMES), height, width), dtype=np.uint8)
    for label, pts in polygons.items():
        cv2.fillPoly(mask[CLASS_TO_IDX[label]], [pts], 1)
    return mask

def resize_with_pad(image, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(resized, top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=0), scale, (top, left)

def resize_mask_with_pad(mask, target_size=1024):
    C, H, W = mask.shape
    scale = target_size / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = np.zeros((C, new_h, new_w), dtype=mask.dtype)
    for c in range(C): resized[c] = cv2.resize(mask[c].astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = np.zeros((C, target_size, target_size), dtype=mask.dtype)
    for c in range(C): padded[c] = cv2.copyMakeBorder(resized[c], top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=0)
    return padded

def find_labeled_samples(labeled_dir):
    samples = []
    if not os.path.exists(labeled_dir): return samples
    for fname in os.listdir(labeled_dir):
        if not fname.lower().endswith('.json'): continue
        json_path = os.path.join(labeled_dir, fname)
        try:
            image_path, polygons = parse_x_anylabeling_json(json_path)
            if not polygons: continue
            if not os.path.exists(image_path):
                base = os.path.splitext(fname)[0]
                for ext in ['.jpg','.jpeg','.png','.JPG','.JPEG','.PNG']:
                    candidate = os.path.join(labeled_dir, base+ext)
                    if os.path.exists(candidate): image_path = candidate; break
            samples.append({'json_path': json_path, 'image_path': image_path, 'polygons': polygons})
        except: pass
    return samples

def split_train_val(samples, train_ratio=0.8, seed=42):
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split_idx = int(len(indices) * train_ratio)
    return [samples[i] for i in indices[:split_idx]], [samples[i] for i in indices[split_idx:]]

def process_and_save(samples, output_dir, phase, target_size):
    save_dir, images_dir, masks_dir = os.path.join(output_dir, phase), os.path.join(output_dir, phase, 'images'), os.path.join(output_dir, phase, 'masks')
    os.makedirs(images_dir, exist_ok=True); os.makedirs(masks_dir, exist_ok=True)
    metadata = []
    for sample in tqdm(samples, desc=f"Processing {phase}"):
        src_img_path, img_name = sample['image_path'], os.path.splitext(os.path.basename(sample['image_path']))[0]
        dst_img_path, dst_mask_path = os.path.join(images_dir, f'{img_name}.jpg'), os.path.join(masks_dir, f'{img_name}.npy')
        img = cv2.imread(src_img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        mask = polygons_to_mask(sample['polygons'], h, w)
        img_resized, scale, pad = resize_with_pad(img_rgb, target_size)
        mask_resized = resize_mask_with_pad(mask, target_size)
        cv2.imwrite(dst_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        np.save(dst_mask_path, mask_resized.astype(np.uint8))
        metadata.append({'image_name': f'{img_name}.jpg', 'image_path': dst_img_path, 'mask_path': dst_mask_path, 'height': target_size, 'width': target_size})
    with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f: json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(metadata)} samples to {save_dir}")

def main():
    import yaml
    with open('config.yaml', 'r') as f: cfg = yaml.safe_load(f)
    labeled_dir, unlabeled_dir, output_dir = cfg['data']['labeled_dir'], cfg['data']['unlabeled_dir'], cfg['data']['processed_dir']
    target_size = cfg['model']['image_size']
    os.makedirs(output_dir, exist_ok=True)
    samples = find_labeled_samples(labeled_dir)
    if not samples: print("No labeled samples found."); return
    train_samples, val_samples = split_train_val(samples, cfg['data']['train_ratio'], cfg['data']['random_seed'])
    process_and_save(train_samples, output_dir, 'train', target_size)
    process_and_save(val_samples, output_dir, 'val', target_size)
    unlabeled_samples = []
    for fname in sorted(os.listdir(unlabeled_dir)):
        if fname.lower().endswith(('.jpg','.jpeg','.png')): unlabeled_samples.append(os.path.join(unlabeled_dir, fname))
    if unlabeled_samples:
        unlabeled_save_dir = os.path.join(output_dir, 'unlabeled')
        os.makedirs(unlabeled_save_dir, exist_ok=True)
        saved = []
        for src_path in tqdm(unlabeled_samples, desc="Resizing unlabeled"):
            img = cv2.imread(src_path)
            if img is None: continue
            dst = os.path.join(unlabeled_save_dir, os.path.splitext(os.path.basename(src_path))[0]+'.jpg')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); img_resized,_,_=resize_with_pad(img,target_size); cv2.imwrite(dst, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            saved.append(dst)
        with open(os.path.join(unlabeled_save_dir, 'filelist.txt'), 'w') as f: f.write('\n'.join(saved) + '\n')
        print(f"Saved {len(saved)} unlabeled images")
    print("Preprocessing complete!")

if __name__ == '__main__': main()