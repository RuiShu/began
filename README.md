# Boundary Equilibrium Generative Adversarial Network

Basic Tensorflow implementation in <= 250 lines of code. 

## Dependencies

1. Tensorflow r0.12 (will update to newest one next week)
2. Tensorbayes
3. Numpy
4. Pillow

## Train

### Data Preparation

Download `img_align_celeba.zip` for the [CelebA dataset](https://drive.google.com/drive/u/0/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8) and unzip to a directory. Then save the 64 x 64 CelebA crops in `.mat` format.

```python
python prepare_celeba_zoom.py --source-dir /path/to/img_align_celeba --dest-dir /path/to/celeba_64_zoom.mat
```

### Train

To train the BEGAN model, execute

```python
python main.py --data-dir /path/to/celeba_64_zoom.mat
```
