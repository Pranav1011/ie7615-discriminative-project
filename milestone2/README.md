# Multi-Object Detection with YOLOv8: IE 7615 Discriminative Project, Milestone 2

IE 7615 Deep Learning for AI, Northeastern University, Fall 2026, Group 3.

**Team:** Hanisha Reddy Cattamanchi Gopinath, Sai Pranav Krovvidi, Mohana Durga Venkata Rohit
Mittapalli, Suhaas Veeramachaneni.

## The project

The class dataset has 73 everyday objects (OBJ001–OBJ073, final Object IDs from the shared
sheet), photographed by the students in single-object photos. In Milestone 1 we classified those
photos. Milestone 2 detects objects: given an image with several of the objects, the model reports
the **Object ID, name and location** of every object in it.

## What we did

1. **Same split as Milestone 1.** We reused the Milestone 1 train/validation/test split (7,315
   photos after removing exact duplicates). Detection images are built only from photos of their
   own split, so nothing leaks into the test set.
2. **Bounding boxes.** The OWLv2 zero-shot detector found the object in each photo, prompted
   with its name. We then checked a random audit of 146 photos, reviewed doubtful
   validation/test boxes one by one, and excluded unreliable ones. That left 6,618 usable photos.
3. **Multi-object images.** We combined randomly selected single-object photos into strict
   grids (1×2 up to 3×3, 2×4 and 4×2), jittered grids and collages, with flip and brightness
   augmentation. In total we generated **4,149 images**:
   - 3,000 for training, 268 for validation and 281 for testing;
   - a 200-image grid stress suite;
   - 4 × 100 member sets (at least 100 per student).
4. **Training.** YOLOv8s, fine-tuned from COCO-pretrained weights on Colab (T4).
   - We trained at 640 px and at 960 px. The 960 px model was chosen on validation, with the
     rule set before training.
   - We also tried a third variant targeting 2×2/3×3 grids without background photos, which
     did not beat it.
5. **Inference.** A standalone `detect.py` (inside the notebook) takes any image and prints
   every object's ID, name, confidence and box, then saves an annotated image.

## Results (final model, held-out test set)

| Metric | Value |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.931 / 0.863 |
| Precision / recall | 0.928 / 0.895 |
| Objects found with the right ID | 90.3% |
| Images with every object right and nothing extra | 60.9% (stress-suite grids: 2×2 60%, 3×3 45%) |

Most errors are missed small objects rather than wrong IDs. On 2×2 and 3×3 grids of photos
drawn from the whole dataset, the model gets 86% and 68% of images fully correct, with 97.0% of
objects identified. Full results and discussion are in the report.

## Running it

Open `Milestone2_Detection.ipynb` in Google Colab with a T4 GPU, then Run all (about 95 minutes).
It expects two zips in your Google Drive (MyDrive):
- the class dataset zip;
- `m2_support.zip`, with the box candidates and review decisions.

The notebook generates all images from fixed seeds, trains, evaluates, and writes an outputs zip
to Drive.
