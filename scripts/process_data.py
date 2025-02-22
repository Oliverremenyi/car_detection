import os
import cv2


def crop_and_save_spots(image_dir, label_dir, output_dir):
    os.makedirs(f"{output_dir}/occupied", exist_ok=True)
    os.makedirs(f"{output_dir}/empty", exist_ok=True)

    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

            image = cv2.imread(image_path)
            image_h, image_w, _ = image.shape

            # read label file
            with open(label_path, "r") as f:
                lines = f.readlines()

            # # process each parking spot
            # for i in lines:
            #     spot_id, x, y, w, h, status = map(float, i.strip().split())
            #     x_px = int(x * image_w)
            #     y_px = int(y * image_h)
            #     w_px = int(w * image_w)
            #     h_px = int(h * image_h)

            for i, line in enumerate(lines):
                spot_id, x, y, w, h = map(float, line.strip().split())

                # Convert normalized coordinates to pixels
                x_px = int(x * image_w)
                y_px = int(y * image_h)
                w_px = int(w * image_w)
                h_px = int(h * image_h)

                # calculate box
                x1 = int(x_px - w_px / 2)
                y1 = int(y_px - h_px / 2)
                x2 = int(x_px + w_px / 2)
                y2 = int(y_px + h_px / 2)

                # crop and save
                spot_image = image[y1:y2, x1:x2]
                class_folder = "occupied" if spot_id == 1 else "empty"
                spot_filename = f"{image_name[:-4]}_spot_{i}.jpg"
                cv2.imwrite(f"{output_dir}/{class_folder}/{spot_filename}", spot_image)


crop_and_save_spots(image_dir="../data/train/images", label_dir="../data/train/labels", output_dir="../data/train_processed")
crop_and_save_spots(image_dir="../data/test/images", label_dir="../data/test/labels", output_dir="../data/test_processed")
crop_and_save_spots(image_dir="../data/valid/images", label_dir="../data/valid/labels", output_dir="../data/valid_processed")

