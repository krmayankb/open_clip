import img2dataset
import argparse
from cloudpathlib import CloudPath

def upload(source, destination):
    img2dataset.download(
        processes_count=16,
        thread_count=128,
        image_size=512,
        resize_mode="keep_ratio_largest",
        url_list = str(source),
        resize_only_if_bigger=True,
        encode_format="jpg",
        output_format="webdataset",
        retries=3,
        enable_wandb=False,
        wandb_project="dataupload",
        skip_reencode=True,
        output_folder=destination,
        input_format="parquet",
        url_col="url",
        caption_col="text",
        number_sample_per_shard=10000,
        distributor="multiprocessing", 
        save_additional_columns=["uid"],
        oom_shard_count=8,
        bbox_col="face_bboxes",
    )
    print("download complete!!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()
    # destination = CloudPath(args.destination)
    upload(args.source, args.destination)

# python3 upload.py --source /mmfs1/gscratch/krishna/mayank/dfn/dfn-medium/metadata_filtered --destination gs://dfn_medium    