My attempt to solve comma.ai programming challenge.
======

The goal of the <a href="https://github.com/commaai/speedchallenge">challenge</a> is to predict the speed of a car from a video. 
Test set MSE: 7.

Main contributions
-----

- Enlarged the data with KITTI odometry data set (less highway, more city drive) and with my data set (with zero translational velocity, pretty easy to label) and rich data augmentation. 
- The model was confused due to other cars' relative movement (the amount of data was not enough to learn it or due to some other reason). Solution: masking out moving objects (cars) using a separate semantic segmentation model during preprocessing. The global mask (for the whole sequence of input images) is obtained by accumulating the sub-masks for every image in the sequence. The same final mask is applied to all images in sequence - therefore, the mask did not move within the sequence.
- Converting i3d model (action classification from video) to the regression one by prediction the continuous value from range (speed) with new train parameter selection using cross-validation.

Model
-----
I used i3d model reported in the paper <a href="https://arxiv.org/abs/1705.07750">"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"</a> and implemented by <a href="https://github.com/piergiaj/pytorch-i3d">piergiaj</a>  using PyTorch. The model accepts as the input a sequence of images.

I converted the initial i3d model from classification to regression by the following steps:
- replacing 400 output channels with the single one.
- changing the kernel size of the last AvgPool3d layer from [2, 7, 7] to [1, 7, 7].
- at the end, I inserted the sigmoid activation function to make the output limited and positive. Finally, I denormalized the output to keep it in the range [0,40] (the max velocity doesn't go over 40 m/s in the current challenge).

Dataset
-----
- The dataset from comma.ai <a href="https://github.com/commaai/speedchallenge">speedchallenge</a> was used.
- To enlarge the data, I added  <a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php">KITTI odometry data set</a>  (the pose labels have been converted to the absolute velocity, and the images were adjusted following the comma data).
- To enlarge the data even more, I recorded several sequences of images of moving cars, holding the camera still (translational velocity is always zero - easy to annotate). It helped to make the data slightly more balanced.

Data preprocessing
-----
- Comma data set. To keep it in accordance with 10 fps KITTI data set in a simpler way (without artificial velocity changes), I converted 20 fps to 10 fps by going through data points with stride 2. Which unfortunately leads to the loss of some data. Initial images with resolution 640x480 have been cropped to 640x320 (keeping 2:1 aspect ratio) and then resized to 224x224(no padding). The resulted images are squeezed horizontally. The images have been split into 85 folders with 120 images in each.
- KITTI data set. Initial images with resolution 1241x376 have been cropped to 752x376 (keeping 2:1 aspect ratio) and then resized to 224x224(no padding). The resulted images are squeezed horizontally. The images have been split into 164 folders with up to 140 images in each.
- My own data set. I made short videos of moving cars with my smartphone, holding the camera still (10 fps). The initial images were grabbed with the resolution 1280x640 (2:1 aspect ratio) and then resized to 224x224(no padding). The resulted images are squeezed horizontally as well. The images have been split into 33 folders with 120 images in each. Probably it was too much data with zero velocity (it was able to learn it pretty well).

During preprocessing, I used a semantic segmentation FPN model to remove cars from the images (done in car_segmantation.ipynb), which was trained by following the default train pipeline proposed by
<a href="https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb">qubvel</a>. It works pretty well even on resized/squeezed images of 224x224 resolution. Before the training of i3d model, I prepared the masks for each frame with removed cars (pixels related to cars were set to 0), which helped to speed up the training. During the training/test time, when I loaded the image sequence for input, I applied masks to all images in sequence in the way I described above (to make the same static mask for all images in sequence).

Train in Colab
-----
- 80/20 train/val split (the first 24 images in every folder are for validation, the rest are for training).
- 10 images in the input sequence.
- Weights init with ImageNet.
- Optimizer: Adam.
- Inital learning rate: 6.0e-06, with 10x reduction of learning rate when validation loss saturated.
- Batch size: 30. The batch size was enlarged even more (actually twice) due to the holding of the optimizer step (update every second iteration).
- 2 output velocity predictions (per sequence) are upscaled to 10 frames linearly. 2 raw predictions per second, far from real-time, but, typically, a car motion model has pretty slow dynamics.

The model was able to show approx. 1.3 MSE on the val data set. Final model chackpoint is uploaded.
![alt text](https://github.com/negvet/speedchallenge/blob/master/train_val_mse_loss_during_trainig.png)

Test
-----
I went through the test images and performed predictions using a sliding window with step 1, and finally, I averaged the velocity for each data frame. The resulting velocity looks pretty smoothed/filtered.
I did not use any additional filters of the predictions.
I did not use any test time augmentation (TTA). Although, sliding window overlap is kind of TTA.
As a result, velocity was predicted for every second frame (for 10 fps case) and then linearly approximated.

Actual test set performance is 7 MSE (provided by comma.ai). The video with printed velocity https://youtu.be/D8Yj8EeJLXQ.

Possible improvements
-----
- Velocity estimation performed poorly during the turns (it is possible to notice during the validation step). KITTY dataset has plenty of turns, but probably not enough. Solution: add additional data with camera rotation and zero translational velocity. Again, it is very easy to label.
- While resizing to 244x244, a significant portion of the information is getting lost due to low resolution. Solution: split the original rectangular image into two squared images, 224x224 (left half and right half), which simulates two camera installation, and more details can be preserved. Predict the velocity independently for every half (we are lucky, and both images have the same velocity). Finally, combine them, ideally using predicted uncertainty, or by taking the arithmetic mean (or weighted average in accordance with the mean validation MSE for each of the half). I conducted several preliminary tests, and the mse was decreased by 5-10% after averaging the velocity between two sub-images (finally, I did not implement it due to the lack of time and GPUs).
- Filter the predictions for higher fps and more reliable results (car motion model needed and ideally the model uncertainty) with KF/EKF for example.
- It can be hard to make a model to predict the uncertainty for both upper improvements (e.g. bayesian NN). The simple workaround is to formulate the velocity prediction as a classification problem (keep the default setting of the i3d model). Split the velocity range [0,40] into 400 intervals/classes and ask a model to predict the class (0.1 m/s range is not that bad). Finally, after softmax activation, we have a probability distribution over the classes, where we can estimate whatever we want, including expectation and variance. I tried it, actually, but I did not have enough resources to tune the training, which is different from the regression case.


