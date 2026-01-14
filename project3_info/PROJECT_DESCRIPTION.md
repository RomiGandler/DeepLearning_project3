# Project 3: Synthetic-to-Real Image Translation for Chessboard Rendering

## Description

In this project, you will train a chessboard image generation model whose goal is to transform synthetic chessboard renders into images that visually match real-world appearance while preserving the geometry and board layout (piece positions, camera pose, etc.).

This is an **image-to-image translation task** focused on closing the visual domain gap between synthetic and real images.

Unlike Project 2, which trains a classifier on synthetic images and evaluates its ability to generalize to real images, Project 3 is not about classification at all. The output here is an **image (regression)**, not a set of labels.

In this project, your goal is to train a model that translates synthetic chessboard renders into realistic-looking images. Once trained, the model will enable you to generate entirely new board states synthetically and then transform them so they appear visually consistent with real data. **(that's the test case in this task)**

## Data Provided

You will receive:
- Labeled frames with the corresponding chessboard state (piece-square positions only, occlusions are not labeled)
- PGN labeled games

The PGN files contain the full game state information, which can be used to easily generate additional frame level labels. *(You have all board states from the PGN throughout the game, but you don't know which state corresponds to each frame.)*

You will receive board-generation code (that I created) using Blender and PyBlender, along with a predefined set of 3D chess pieces and a chessboard. If you need additional synthetic variation (e.g., different piece designs), you may adapt the code and download more Blender chess sets (a website will be provided).

**You need to install Blender to your computer:** https://www.blender.org/download/

## Key Components

- **Data Generation:** Using provided board generation code (created using Blender and PyBlender), you will need to generate the data you need.

- **Model Training:** Train a model that converts synthetic chessboard renders into realistic-looking images while preserving geometric structure: piece identity, board layout, etc.

- **Methods:** Explore image-to-image translation methods (e.g., GANs, CycleGAN, diffusion-based models, etc.)

- **Evaluation:** Evaluate how well the translated images match the real domain visually (qualitatively and/or quantitatively)

- **Occlusion Handling:** Some of the images have occluded squares (see examples in PDF of project 1). Is it possible to generate data without occlusion?

- **Generalization:** Robustness to new game states.

## Additional Notes

1. **PGN Usage:** You don't have to use the PGN games, but you might find it useful.

2. **Temporal Information (Training):** You are allowed to leverage the temporal nature of the video (i.e., consecutive frames) to generate additional labeled data from the PGN files.

3. **Labeling Approaches:** For example, you can do so either by manually labeling frames or by applying any algorithm of your choice:
   - Classical methods
   - Self-/unsupervised approaches
   - Your own trained models
   - Off-the-shelf models that will label your data

4. **Model Input/Output (Inference):** Your generation model must **NOT** rely on temporal information (e.g., what happened in previous or subsequent frames). Its only input is a **single static image** of the board, and its output is a realistic image.

5. **Generalization:** Try to make your model generalize to new game states the best you can. Be creative, train the model smartly.

6. **Additional Assets:** You are allowed to download more 3D Chess assets, adjust my code to match it for rendering, and use it as more diverse synthetic data of chess boards and pieces if you find it useful.
   - Example: https://www.blendswap.com/blend/29244

## Data Structure

### Example:
You will be provided with ZIP files for each game.

Each ZIP contains:
- `images/` folder
- Either a `game.csv` file or a PGN file with the corresponding labels

## Project Goals Summary

1. ✅ Generate synthetic chess board images using Blender
2. ✅ Train an image-to-image translation model (synthetic → realistic)
3. ✅ Preserve board geometry and piece positions
4. ✅ Handle occlusions appropriately
5. ✅ Generalize to new, unseen board states
6. ✅ Evaluate visual quality of translated images

