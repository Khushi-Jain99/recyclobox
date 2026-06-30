import os
import sys
import shutil
import random

# Add parent directory of preprocessing folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config_helper import load_config

CLASS_MAPPINGS = {
    'hazardous': [
        'batteries',
        'e-waste',
        'paints',
        'pesticides'
    ],
    'non_recyclable': [
        'ceramic_product',
        'diapers',
        'sanitary_napkin',
        'stroform_product'
    ],
    'organic': [
        'coffee_tea_bags',
        'egg_shells',
        'food_scraps',
        'kitchen_waste',
        'yard_trimmings'
    ],
    'plastic': [
        'plastic_bottles',
        'platics_bags_wrappers'
    ],
    'paper': [
        'paper_products'
    ],
    'glass': [
        'glass_containers'
    ],
    'metal': [
        'cans_all_type'
    ]
}

def clean_processed_dir(processed_dir):
    if os.path.exists(processed_dir):
        print(f"Cleaning existing processed directory: {processed_dir}")
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

def main():
    config = load_config()
    raw_dir = config['dataset']['raw_dir']
    processed_dir = config['dataset']['processed_dir']
    train_split = config['dataset']['train_split']
    val_split = config['dataset']['val_split']
    test_split = config['dataset']['test_split']
    seed = config['dataset']['random_seed']
    
    random.seed(seed)
    
    clean_processed_dir(processed_dir)
    
    print("Preparing and splitting dataset...")
    
    summary_stats = {}
    
    for target_class, source_folders in CLASS_MAPPINGS.items():
        class_images = []
        for folder in source_folders:
            folder_path = os.path.join(raw_dir, folder)
            if not os.path.exists(folder_path):
                folder_path = os.path.join(raw_dir, folder.replace('/', '\\'))
            
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            class_images.append(os.path.join(root, file))
            else:
                print(f"Warning: Source folder {folder_path} does not exist.")
        
        if not class_images:
            print(f"Warning: No images found for class '{target_class}'")
            continue
            
        random.shuffle(class_images)
        
        total_images = len(class_images)
        train_end = int(total_images * train_split)
        val_end = train_end + int(total_images * val_split)
        
        train_imgs = class_images[:train_end]
        val_imgs = class_images[train_end:val_end]
        test_imgs = class_images[val_end:]
        
        summary_stats[target_class] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs),
            'total': total_images
        }
        
        splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        for split_name, img_list in splits.items():
            split_class_dir = os.path.join(processed_dir, split_name, target_class)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for idx, img_path in enumerate(img_list):
                ext = os.path.splitext(img_path)[1]
                new_filename = f"{target_class}_{idx:05d}{ext}"
                dest_path = os.path.join(split_class_dir, new_filename)
                shutil.copy2(img_path, dest_path)
                
    # Print summary statistics
    print("\n" + "="*50)
    print("Dataset Preparation Summary")
    print("="*50)
    print(f"{'Class':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-"*50)
    grand_total = {'train': 0, 'val': 0, 'test': 0, 'total': 0}
    for cls, stats in summary_stats.items():
        print(f"{cls:<15} | {stats['train']:<7} | {stats['val']:<7} | {stats['test']:<7} | {stats['total']:<7}")
        for key in grand_total:
            grand_total[key] += stats[key]
    print("-"*50)
    print(f"{'Total':<15} | {grand_total['train']:<7} | {grand_total['val']:<7} | {grand_total['test']:<7} | {grand_total['total']:<7}")
    print("="*50)

if __name__ == "__main__":
    main()
