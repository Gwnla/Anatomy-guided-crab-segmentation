import os, sys, argparse, yaml, cv2, numpy as np, torch, matplotlib.pyplot as plt
from train import PartSegmentModel, CLASS_NAMES, CLASS_COLORS

def load_model(checkpoint_path, config_path='config.yaml'):
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PartSegmentModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('student_state_dict', ckpt))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model, cfg, device

def resize_with_pad(image, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(resized, top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=0), scale, (top, left)

def preprocess_image(image_path, target_size=1024):
    image = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Cannot read: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    padded, scale, pad = resize_with_pad(image_rgb, target_size)
    normalized = padded.astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float(), image_rgb, original_h, original_w, scale, pad

def postprocess_mask(pred_probs, original_h, original_w, scale, pad):
    C, H, W = pred_probs.shape
    top, left = pad
    if H > 0 and W > 0:
        unpad_h, unpad_w = int(original_h * scale), int(original_w * scale)
        pred_probs = pred_probs[:, top:top+unpad_h, left:left+unpad_w]
    masks = np.stack([cv2.resize(pred_probs[c], (original_w, original_h), interpolation=cv2.INTER_LINEAR) for c in range(C)], axis=0)
    return masks

def compute_morphological_metrics(masks):
    C, H, W = masks.shape
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = (masks[i] > 0.5).astype(np.uint8) * 255
        if mask.sum() == 0:
            metrics[f'{name}_area'] = 0.0; metrics[f'{name}_perimeter'] = 0.0; metrics[f'{name}_aspect_ratio'] = 0.0; continue
        area_pixels = np.sum(mask > 0); metrics[f'{name}_area'] = float(area_pixels)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours); metrics[f'{name}_perimeter'] = float(perimeter)
        if len(contours) > 0:
            all_pts = np.vstack(contours); rect = cv2.minAreaRect(all_pts); w_rect, h_rect = rect[1]
            metrics[f'{name}_aspect_ratio'] = max(w_rect, h_rect) / min(w_rect, h_rect) if w_rect > 0 and h_rect > 0 else 1.0
        else: metrics[f'{name}_aspect_ratio'] = 1.0
    for pair, key in [((1, 2), 'claw_symmetry_index'), ((3, 4), 'legs_symmetry_index')]:
        a = metrics.get(f'{CLASS_NAMES[pair[0]]}_area', 0); b = metrics.get(f'{CLASS_NAMES[pair[1]]}_area', 0)
        metrics[key] = 1.0 - abs(a - b) / (a + b + 1e-6) if (a + b) > 0 else 1.0
    metrics['overall_symmetry_index'] = (metrics['claw_symmetry_index'] + metrics['legs_symmetry_index']) / 2
    return metrics

def overlay_mask_on_image(image, mask, alpha=0.5):
    overlay = image.copy()
    fg = mask.sum(axis=0) > 0.5
    class_map = np.argmax(mask, axis=0).astype(np.int32); class_map[~fg] = -1
    for c in range(len(CLASS_NAMES)):
        color = CLASS_COLORS[c]; pm = class_map == c
        if pm.any(): overlay[pm] = (overlay[pm].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return overlay

def predict_single(image_path, model, cfg, device, save_dir=None, use_tta=True):
    tensor, original_img, orig_h, orig_w, scale, pad = preprocess_image(image_path, cfg['model']['image_size'])
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        if use_tta:
            logits_flip = model(tensor.flip(-1)).flip(-1)[:, [0, 2, 1, 4, 3]]
            logits = (logits + logits_flip) * 0.5
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    masks = postprocess_mask(probs, orig_h, orig_w, scale, pad)
    binary_masks = (masks > 0.5).astype(np.uint8)
    metrics = compute_morphological_metrics(binary_masks)
    overlay = overlay_mask_on_image(original_img, binary_masks)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    px_per_cm = 194.9
    results = {'image_name': os.path.basename(image_path), 'original_size': (orig_h, orig_w), 'metrics': metrics}
    for n in CLASS_NAMES:
        results[f'cm_area_{n}'] = metrics[f'{n}_area'] / (px_per_cm ** 2)
        results[f'cm_perimeter_{n}'] = metrics[f'{n}_perimeter'] / px_per_cm

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        fig = plt.figure(figsize=(20, 10), facecolor='white')
        fig.suptitle('Crab Part Segmentation', fontsize=26, fontweight='bold', color='#222', y=0.97)
        ax_img = fig.add_axes([0.02, 0.05, 0.55, 0.85])
        ax_img.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); ax_img.axis('off')
        ax_info = fig.add_axes([0.60, 0.05, 0.38, 0.85]); ax_info.axis('off'); ax_info.set_facecolor('white')
        ax_info.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax_info.transAxes, color='#2C3E50', zorder=0))
        ax_info.text(0.5, 0.94, 'Measurement Report', fontsize=18, color='white', transform=ax_info.transAxes, fontweight='bold', ha='center', va='center')
        color_map = {'body':'#E74C3C','left_claw':'#27AE60','right_claw':'#2980B9','left_legs':'#F39C12','right_legs':'#8E44AD'}
        y_pos = 0.82
        ax_info.text(0.05, y_pos, 'Part', fontsize=16, color='#333', transform=ax_info.transAxes, fontweight='bold')
        ax_info.text(0.70, y_pos, 'Area (cm' + '\xb2' + ')', fontsize=14, color='#333', transform=ax_info.transAxes, fontweight='bold', ha='right')
        ax_info.text(0.95, y_pos, 'Perimeter (cm)', fontsize=14, color='#333', transform=ax_info.transAxes, fontweight='bold', ha='right')
        y_pos -= 0.05
        ax_info.plot([0.03, 0.97], [y_pos, y_pos], color='#ccc', linewidth=0.8, transform=ax_info.transAxes); y_pos -= 0.06
        for name in CLASS_NAMES:
            ac = metrics[f'{name}_area'] / (px_per_cm ** 2); pc = metrics[f'{name}_perimeter'] / px_per_cm
            ax_info.text(0.05, y_pos, name.replace('_', ' ').title(), fontsize=15, color=color_map[name], transform=ax_info.transAxes, fontweight='bold')
            ax_info.text(0.70, y_pos, f"{ac:.1f}", fontsize=15, color='#333', transform=ax_info.transAxes, fontweight='bold', ha='right')
            ax_info.text(0.95, y_pos, f"{pc:.1f}", fontsize=15, color='#333', transform=ax_info.transAxes, fontweight='bold', ha='right'); y_pos -= 0.09
        y_pos -= 0.02
        ax_info.plot([0.03, 0.97], [y_pos, y_pos], color='#ccc', linewidth=0.8, transform=ax_info.transAxes); y_pos -= 0.06
        for label, key in [('Claw Symmetry', 'claw_symmetry_index'), ('Leg Symmetry', 'legs_symmetry_index'), ('Overall Symmetry', 'overall_symmetry_index')]:
            ax_info.text(0.05, y_pos, label, fontsize=15, color='#333', transform=ax_info.transAxes)
            ax_info.text(0.95, y_pos, f"{metrics[key]:.3f}", fontsize=16, color='#333', transform=ax_info.transAxes, fontweight='bold', ha='right'); y_pos -= 0.08
        save_path = os.path.join(save_dir, f'{base_name}_result.png')
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white'); plt.close(fig)
        # Individual parts figure
        fig_p, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig_p.suptitle('Individual Part Segmentation', fontsize=20, fontweight='bold', y=1.02)
        for i, name in enumerate(CLASS_NAMES):
            po = original_img.copy(); mc = binary_masks[i]; c = np.array(CLASS_COLORS[i], dtype=np.uint8)
            po[mc > 0] = (po[mc > 0].astype(np.float32) * 0.4 + c.astype(np.float32) * 0.6).astype(np.uint8)
            axes[i].imshow(po); axes[i].set_title(name.replace('_', ' ').title(), fontsize=14, fontweight='bold', color=color_map[name]); axes[i].axis('off')
        parts_path = os.path.join(save_dir, f'{base_name}_parts.png')
        fig_p.savefig(parts_path, dpi=180, bbox_inches='tight', facecolor='white'); plt.close(fig_p)
        print(f"Result: {save_path}")

    print("\n=== Measurements ===")
    for name in CLASS_NAMES:
        ac = metrics[f'{name}_area'] / (px_per_cm ** 2); pc = metrics[f'{name}_perimeter'] / px_per_cm
        print(f"  {name}: Area={ac:.1f} cm2  Perimeter={pc:.1f} cm")
    print(f"  Claw Symmetry: {metrics['claw_symmetry_index']:.4f}  Leg Symmetry: {metrics['legs_symmetry_index']:.4f}  Overall: {metrics['overall_symmetry_index']:.4f}")
    return results

def predict_batch(image_dir, model, cfg, device, save_dir=None, use_tta=True):
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(valid_exts)]
    all_results = []
    for fname in image_files:
        print(f"\nProcessing: {fname}")
        all_results.append(predict_single(os.path.join(image_dir, fname), model, cfg, device, save_dir, use_tta))
    if save_dir and all_results:
        import csv
        combined = []
        for r in all_results:
            entry = {'image_name': r['image_name']}; entry.update(r['metrics'])
            for k, v in r.items(): entry[k] = v if not isinstance(v, dict) else None
            combined.append(entry)
        csv_path = os.path.join(save_dir, 'all_measurements.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if combined: csv.DictWriter(f, fieldnames=combined[0].keys()).writeheader(); csv.DictWriter(f, fieldnames=combined[0].keys()).writerows(combined)
        print(f"\nCombined: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Crab Part Segmentation Predict')
    parser.add_argument('input', type=str, nargs='?', default='./test_images', help='Path to image or directory')
    parser.add_argument('--checkpoint', type=str, default='./outputs/best_model.pth', help='Model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--save_dir', type=str, default='./outputs/predictions', help='Save directory')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        os.makedirs(args.input, exist_ok=True); print(f"Put images in '{args.input}' and run again."); return
    model, cfg, device = load_model(args.checkpoint, args.config)
    if os.path.isfile(args.input): predict_single(args.input, model, cfg, device, args.save_dir, not args.no_tta)
    else: predict_batch(args.input, model, cfg, device, args.save_dir, not args.no_tta)

if __name__ == '__main__': main()