# Plan: Synthetic-to-Real Chessboard Image Translation

The goal is to bridge the visual "domain gap" between synthetic 3D renders and real-world photographs of chess boards.

## 1. Data Preparation and Mapping

- **Map Real Data**: Link real images from `Labeled Chess data...` to their FEN strings provided in the corresponding `.csv` files.
- **Generate Synthetic Pairs**: Create a script to run Blender (using `blender/chess_position_api_v2.py`) for every FEN in the real dataset.
- **Data Augmentation (Optional)**: If more training data is needed, use the provided PGN files to generate additional board states and render them synthetically to expand Domain A.
- **Match Perspective**: Identify which predefined view (`overhead`, `east`, `west`) most closely matches the camera angle in the real games to ensure the translation model focuses on texture rather than geometric correction.

## 2. Dataset Organization

- Structure the data for an unpaired image-to-image translation model (e.g., CycleGAN or CUT).
- Create two domains:
    - **Domain A (Synthetic)**: Renders from Blender.
    - **Domain B (Real)**: Photos from the provided game frames.
- Standardize image sizes (e.g., 256x256 or 512x512) for training.

## 3. Model Implementation (PyTorch)

- **Architecture**: Use a CycleGAN or Contrastive Unpaired Translation (CUT) architecture. CUT is recommended as it often preserves geometric structure better than standard CycleGAN.
- **Generator**: A ResNet-based or U-Net-based generator to transform images.
- **Discriminator**: A PatchGAN discriminator to ensure local texture realism.

## 4. Training on Colab

- Export the prepared dataset (Synthetic and Real images) to Google Drive.
- Implement the training loop in a Jupyter Notebook.
- Monitor loss (Adversarial, Cycle-consistency/Contrastive) and visualize intermediate results.

## 5. Evaluation and Inference

- **Test Set**: Generate renders for FENs not seen during training.
- **Inference Script**: A script that takes a new FEN, renders it via Blender, and passes it through the trained Generator to produce a "realistic" image.
- **Visual Validation**: Compare the output against real images for consistency in lighting, shadows, and piece textures.

## 6. Handling Occlusions

- Analyze if the model should learn to remove occlusions (like hands) or if synthetic data should be augmented with "fake" occluded regions to help the model ignore them.