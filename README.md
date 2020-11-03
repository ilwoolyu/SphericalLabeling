# SphericalLabeling: Cortical Surface Labeling using Spherical Data Augmentation and Context-aware Training

## Description
We adapt [spherical convolutional neural networks](https://github.com/maxjiang93/ugscnn) designed for generic semantic segmentation tasks. Yet, direct use of the generic networks for our sulcal labeling task is challenging due to (1) limited training samples in capturing anatomical variability and (2) lack of hierarchical neuroanatomical association. To adapt the generic networks to our problem setting, we propose spherical data augmentation as well as context-aware training to efficiently utilize existing training samples and to accept a wide range of individual variability by considering training data synthesis and contextual information.
![image](https://user-images.githubusercontent.com/9325798/97788576-faf7a280-1b87-11eb-8e75-38670b9dda14.png)

## Singularity images [[link](https://vanderbilt.box.com/v/SphericalLabeling)]
* Cortical sulci
  * Lateral Prefrontal Cortex [[paper1](#ref3)][[paper2](#ref4)]

* Cortical parcellation
  * BrainCOLOR [[paper](#ref2)][[protocol](https://mindboggle.info/braincolor)]
  * DKT31 [[paper](#ref2)][[protocol](https://mindboggle.readthedocs.io/en/latest/labels.html)]

## Requirements
* [HSD](https://github.com/ilwoolyu/HSD): surface registration (available at [Docker](https://hub.docker.com/r/ilwoolyu/cmorph) and [Singularity](https://vanderbilt.box.com/v/cmorph-release))
* [SphericalRemesh](https://github.com/ilwoolyu/SphericalRemesh): data augmentation (available at [Docker](https://hub.docker.com/r/ilwoolyu/cmorph) and [Singularity](https://vanderbilt.box.com/v/cmorph-release))
* pytorch (>=`0.4.0`)
* numpy (>=`1.11.3`)
* scipy (>=`1.2.1`)

## Data structure
For a better processing flow, we assume the FreeSurfer standard data structure.
```
$SUBJECTS_DIR
├── subj_1
│   └── surf
│       ├── lh.curv
│       ├── lh.inflated.H
│       ├── lh.sulc
│       └── lh.sphere
.
.
└── subj_N
    └── surf
        ├── lh.curv
        ├── lh.inflated.H
        ├── lh.sulc
        └── lh.sphere
```
We will create `$DATA_DIR` that ultimately contains re-tessellated spherical data by a certain level of icosahedral subdivision. Be sure to create `$DATA_DIR/features` and `$DATA_DIR/labels` before the processing.
```
$DATA_DIR
├── features
│   ├── subj_1.lh.aug0.curv.dat
│   ├── subj_1.lh.aug0.iH.dat
│   ├── subj_1.lh.aug0.sulc.dat
.
.
│   ├── subj_N.lh.aug15.curv.dat
│   ├── subj_N.lh.aug15.iH.dat
│   ├── subj_N.lh.aug15.sulc.dat
.
.
│   ├── subj_N.lh.aug0.curv.dat
│   ├── subj_N.lh.aug0.iH.dat
│   ├── subj_N.lh.aug0.sulc.dat
.
.
│   ├── subj_N.lh.aug15.curv.dat
│   ├── subj_N.lh.aug15.iH.dat
│   └── subj_N.lh.aug15.sulc.dat
└── labels
    ├── subj_1.lh.aug0.label.dat
    ├── subj_1.lh.aug1.label.dat
    .
    .
    ├── subj_N.lh.aug14.label.dat
    └── subj_N.lh.aug15.label.dat
```
## Step 0. Environment setup
```
$ git clone https://github.com/ilwoolyu/SphericalLabeling.git
```

Run `python gen_ico.py` to generate icosahedral mesh files (`pkl` and `vtk`). By default, they will be generated in `$PWD/mesh_files`.

## Step 1. Surface registration
Run HSD registration with `--outputcoeff` (see how to register spheres using HSD in a [group-wise](https://github.com/ilwoolyu/HSD#usage) or [pair-wise](https://github.com/ilwoolyu/HSD#pairwise-registration) manner). For quick group-wise registration on FreeSurfer, use [this script](https://github.com/ilwoolyu/HSD/tree/master/script). After HSD, the spherical harmonics coefficients (`txt`) will be generated. For convenience, assume that `HSD` stores the spherical harmonics coefficients for each subject directory. 

Then, let's convert FreeSurfer data into `txt` files (values per line). Using `cut` (default shell command) and `mris_convert` (FreeSurfer command), surface geometry files can be converted into the ASCII format. For example, the following command converts `lh.curv`:
```
$ mris_convert \
-c $SUBJECTS_DIR/surf/lh.curv \
$SUBJECTS_DIR/surf/lh.sphere \
$SUBJECTS_DIR/HSD/lh.curv.asc

$ cut -d ' ' -f 5 $SUBJECTS_DIR/HSD/lh.curv.asc > $SUBJECTS_DIR/HSD/lh.curv.txt

$ rm $SUBJECTS_DIR/HSD/lh.curv.asc
```
By slightly modifying the script above, the FreeSurfer geometry files can be converted in a consistent manner below:
* `lh.curv` --> `lh.curv.txt`
* `lh.inflated.H` --> `lh.iH.txt`
* `lh.sulc` --> `lh.sulc.txt`

Same stroy for `lh.label.txt`. Any non-negative integer ranges are allowed for label definitions. Be sure the files to have row-wise label information. Finally, convert `lh.sphere` into a `VTK` format:
```
$ mris_convert $SUBJECTS_DIR/surf/lh.sphere $SUBJECTS_DIR/HSD/lh.sphere.vtk
```
Loop through all subjects for data preparation. They will be later used for data augmentation. After conversion, the directory structure should look like

```
$SUBJECTS_DIR
├── subj_1
│   ├── HSD
│   |   ├── lh.curv.txt
│   |   ├── lh.coeff.txt
│   |   ├── lh.iH.txt
│   |   ├── lh.label.txt
│   |   ├── lh.sulc.txt
│   |   └── lh.sphere.vtk
│   └── surf
.
.
└── subj_N
    ├── HSD
    |   ├── lh.curv.txt
    |   ├── lh.coeff.txt
    |   ├── lh.iH.txt
    |   ├── lh.label.txt
    |   ├── lh.sulc.txt
    |   └── lh.sphere.vtk
    └── surf
```

## Step 2. Spherical data augmentation
Run [SphericalRemesh](https://github.com/ilwoolyu/SphericalRemesh) to generate intermediate deformation. For each individual subject, the intermediate deformation can be generated by the following command:
```
$ SphericalRemesh \
-s $SUBJECTS_DIR/subj_1/HSD/lh.sphere.vtk \
-r $PWD/mesh_files/icosphere_5.vtk \
-c $SUBJECTS_DIR/subj_1/HSD/lh.coeff.txt \
-p $SUBJECTS_DIR/subj_1/HSD/lh.curv.txt \
   $SUBJECTS_DIR/subj_1/HSD/lh.iH.txt \
   $SUBJECTS_DIR/subj_1/HSD/lh.sulc.txt \
--outputProperty $DATA_DIR/features/subj_1.lh.aug \
--deg0 0 \
--deg 15 \
--ftype double
```
Similarly, label data can be augmented as well. Be sure to enable `--nneighbor` flag to avoid barycentric interpolation of categorical data.
```
$ SphericalRemesh \
-s $SUBJECTS_DIR/subj_1/HSD/lh.sphere.vtk \
-r $PWD/mesh_files/icosphere_5.vtk \
-c $SUBJECTS_DIR/subj_1/HSD/lh.coeff.txt \
-p $SUBJECTS_DIR/subj_1/HSD/lh.label.txt \
--outputProperty $DATA_DIR/labels/subj_1.lh.aug \
--deg0 0 \
--deg 15 \
--ftype int16 \
--nneighbor
```
> Note: `icosphere_#.vtk` ranges 0 to 7. `#` will decide the maximum resolution of input channels in the networks. For `#=5`, you can control input channel resolution (`--max_level` in `train.py`) up to 5.

Once again, loop through all subjects for training sample augmentation.

## Step 3. Training
Run `train.sh` to train the data. You need to setup `train.sh` properly, including `$DATA_DIR`, output directory, hyper-parameters, etc. The script supports k-fold cross-validation. You can customize k by `--kcv` (5 by default) and set the current fold by `--fold`. If successful, you will see the training log like
```
$ ./train.sh
Namespace(batch_size=4, blackout_id='', classes=[0, 1, 2, 3, 4, 6, 7, 8, 10], data_folder='/home/user/data', decay=False, deg=15, drop=[0], epochs=30, feat=32, fold=1, hemi='lh', in_ch=['curv', 'iH', 'sulc', 'label'], kcv=5, log_dir='/home/user/data/cv/fold1', log_interval=300, lr=0.01, max_level=5, mesh_folder='./mesh_files', min_level=0, no_cuda=False, optim='adam', resume=None, seed=1, test_batch_size=10, train_stats_freq=0, ts=10
5179337 paramerters in total
Train Epoch: 1 [0/1200 (0%)] Loss: 2.247794
[Epoch 1 train stats]: Avg loss: 0.2559
[Epoch 1 val stats]: MIoU: 0.3506; Mean Accuracy: 0.4789; Mean Dice: 0.5035; Avg loss: 0.0841
[Epoch 1 test stats]: MIoU: 0.3543; Mean Accuracy: 0.4814; Mean Dice: 0.4840; Avg loss: 0.0869
...
```
> Note: the classes do not necessarily contain all the labels in `label.dat`. For the training, you may want to choose labels of interests only.

Once trained, you will find trained models '.pth' in each fold.

## Step 4. Inference
### Within cross validation
In the cross validation, `aug0.dat` files (rigid rotation) are used for validation and test. They can be thus used for the inference. Copy them to `$TEST_DIR`:
```
$ cp $DATA_DIR/features/subj_1.lh.aug0.*.dat $TEST_DIR
```
To infer labels, run the following (GPU-free) command:
```
$ ./test.sh $PATH_TO_PTH $TEST_DIR
```
>`test.py` supports multiple formats (`txt`, `dat`, `mat`). Use `--fmt` to determine the format, depending on applications.

This will generate raw values from the output layer per label as `$TEST_DIR/subj_1.?h.aug0.prob*.txt`. After applying normalization such as soft-max on these values, you can either find maximum likelihood or employ a graph-cut technique to determine final labels.

In case of mapping `prob*.txt` to the native sphere, the rotated sphere used for `aug0` is needed. To generate rigid rotation, run:
```
$ SphericalRemesh \
-s $SUBJECTS_DIR/subj_1/HSD/lh.sphere.vtk \
-c $SUBJECTS_DIR/subj_1/HSD/lh.coeff.txt \
--deform $TEST_DIR/subj_1.lh.sphere.rigid.vtk \
--deg 0
```
Then, use `$TEST_DIR/subj_1.lh.sphere.rigid.vtk` to re-tessellate `prob*.txt` with the native sphere:
```
$ SphericalRemesh \
-s $PWD/mesh_files/icosphere_5.vtk \
-r $TEST_DIR/subj_1.lh.sphere.rigid.vtk \
-p $TEST_DIR/subj_1.?h.aug0.prob1.txt $TEST_DIR/subj_1.?h.aug0.prob2.txt ... \
--outputProperty $TEST_DIR/subj_1.lh
```
This will generate `$TEST_DIR/subj_1.lh.prob*.txt` tessellated on the native sphere.

### Outside of cross validation
Generate `txt` files first (see Step 1). Similar to Step 2, re-tessellated spherical data (`dat` files) are needed as the networks are trained with icosahedral mesh. To re-tessellate spherical data, run the following command:
```
$ SphericalRemesh \
-s $SUBJECTS_DIR/subj_1/HSD/lh.sphere.vtk \
-r $PWD/mesh_files/icosphere_5.vtk \
-p $SUBJECTS_DIR/subj_1/HSD/lh.curv.txt \
   $SUBJECTS_DIR/subj_1/HSD/lh.iH.txt \
   $SUBJECTS_DIR/subj_1/HSD/lh.sulc.txt \
--outputProperty $TEST_DIR/subj_1.lh.res \
--deg 0 \
--ftype double
```
This will generate 
```
$TEST_DIR
├── subj_1.lh.res.curv.dat
├── subj_1.lh.res.iH.dat
└── subj_1.lh.res.sulc.dat
```
Once again, run the following command:
```
$ ./test.sh $PATH_TO_PTH $TEST_DIR
```
As described already, use a proper method to determine final labels from the generated `prob`. If needed, map them to the native sphere before determining the final labels (see the sub-section above).
## Context-aware training
If neuroanatomical association is known (e.g., hierarchical emergence of sulci - non-tertiary: first; tertiary: last), we employ that information for better guidance of labeling. The implementation is straightforward. First, training can be done with upper levels of the hierarchy. Use `--classes` flag in `train.py`.

Once trained, the raw outputs of the inference from the networks can be generated by Step 4. Use `--fmt dat` flag in `test.py` for all training samples. This will generate the binary format of the raw outputs (`prob?.dat`) ready for the next stage of the training. By adding `prob*` to `--in_ch` and updating `--classes` in `train.py`, the second stage of the training starts. Depending on a total of levels in the hierarchy, the steps repeat as needed. See more details in [4](#ref4).

## References
Please cite the following papers if you find it useful.
* <a id="ref1"></a>Lyu, I., Kang, H., Woodward, N., Styner, M., Landman, B., <a href="https://doi.org/10.1016/j.media.2019.06.013">Hierarchical Spherical Deformation for Cortical Surface Registration</a>, <i>Medical Image Analysis</i>, 57, 72-88, 2019</li>
* <a id="ref2"></a>Parvathaneni, P., Bao, S., Nath, V., Woodward, N., Claassen, D., Cascio, C., Zald, D., Huo, Y., Landman, B., Lyu, I., <a href="https://doi.org/10.1007/978-3-030-32248-9_56">Cortical Surface Parcellation using Spherical Convolutional Neural Networks</a>, <i>Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2019</i>. LNCS11766, 501-509, 2019.
* <a id="ref3"></a>Hao, L., Bao, S., Tang, Y., Gao, R., Parvathaneni, P., Miller, J., Voorhies, W., Yao, J., Bunge, S., Weiner, K., Landman, B., Lyu, I., <a href="https://doi.org/10.1109/ISBI45749.2020.9098414">Automatic Labeling of Cortical Sulci Using Spherical Convolutional Neural Networks in a Developmental Cohort</a>, <i>IEEE International Symposium on Biomedical Imaging (ISBI) 2020</i>, IEEE, 412-415, 2020.
* <a id="ref4"></a>Lyu I., Bao, S., Hao, L., Yao, J., Miller, J., Voorhies, W., Taylor, W., Bunge, S., Weiner, K., Landman, B., Labeling Lateral Prefrontal Sulci using Spherical Data Augmentation and Context-aware Training, under review.