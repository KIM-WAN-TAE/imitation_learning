import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dual_arm.datasets.lerobot_dataset import CODEBASE_VERSION
from dual_arm.datasets.utils import create_dual_arm_dataset_card

def upload_datasets(username):
    api = HfApi()
    base_path = Path("/home/wt/dual_arm/src/dual_arm/pickandplace")
    
    # List to store successfully pushed repo IDs
    pushed_repo_ids = []
    
    dataset_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    print(f"Found {len(dataset_dirs)} dataset directories.")
    
    for ds_dir in dataset_dirs:
        # Check if it looks like a LeRobot dataset
        info_path = ds_dir / "meta" / "info.json"
        if not info_path.exists():
            print(f"Skipping {ds_dir.name} as it doesn't look like a valid LeRobot dataset (missing meta/info.json).")
            continue
            
        repo_id = f"{username}/{ds_dir.name}"
        print(f"Uploading {ds_dir.name} to {repo_id}...")
        
        try:
            # Create repo if it doesn't exist
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            
            # Upload the whole folder
            api.upload_folder(
                folder_path=str(ds_dir),
                repo_id=repo_id,
                repo_type="dataset",
            )
            
            # Create the required codebase version tag (e.g., v3.0)
            try:
                api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
                print(f"Created tag {CODEBASE_VERSION} for {repo_id}")
            except Exception as tag_e:
                print(f"Note: Tag {CODEBASE_VERSION} might already exist or failed: {tag_e}")
            
            # Create and push dataset card
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                
                card = create_dual_arm_dataset_card(
                    dataset_info=info,
                    tags=["dual_arm"],
                )
                card.push_to_hub(repo_id=repo_id, repo_type="dataset")
            except Exception as card_e:
                print(f"Warning: Could not create/push dataset card for {ds_dir.name}: {card_e}")

            pushed_repo_ids.append(repo_id)
            print(f"Successfully pushed {repo_id}")
            
        except Exception as e:
            print(f"Failed to push {ds_dir.name}: {e}")

    # Save the list to a JSON file
    output_file = Path("dataset_repo_ids.json")
    with open(output_file, "w") as f:
        json.dump(pushed_repo_ids, f)
    
    print(f"\n--- DONE ---")
    print(f"Saved {len(pushed_repo_ids)} repo IDs to {output_file.absolute()}")
    print("\nTo train with these datasets, use the following argument:")
    repo_list_str = str(pushed_repo_ids).replace(" ", "")
    print(f"dataset.repo_id=\"{repo_list_str}\"")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True, help="HF Hub Username")
    args = parser.parse_args()
    
    upload_datasets(args.username)
