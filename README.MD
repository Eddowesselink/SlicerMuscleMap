# SlicerMuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle

SlicerMuscleMap is an open-source segmentation model for the automated analysis of whole-body muscle morphometry and composition.

![][MuscleMap_Slicer3D.png]

When citing MuscleMap, please cite the following publication:

> McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. J Imaging. 2024;10(11):262. <https://doi.org/10.3390/jimaging10110262> 

> Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, Weber II KA.Convolutional neural networks for the automatic segmentation of lumbar paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. <https://doi.org/10.1038/s41598-022-16710-5>

##  Set up

If you have a GPU available and installed then whole-body segmentation can be computed in a few minutes. Therefore, it is recommended to set up GPU acceleration as described in this section. CUDA installation is not necessary when a suitable GPU is not available, but will   take more time. By default, the Slicer3D Pytorch module will be installed alongside the MuscleMap installation.

**To use a GPU,** you need a compatible NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) installed or a compatible AMD GPU with [ROCM](https://www.amd.com/en/products/software/rocm.html) installed. You then need to install the corresponding GPU-compatible version of [PyTorch v2.4.0](https://pytorch.org/get-started/previous-versions/). 


### Usage
1. Install important MuscleMap depencies by clicking on the button in the main module.
2. Load an imput image into the Slicer3D console.
3. Click on 'Run MuscleMap Segmentation'. Dependent on the availability of a GPU and image size, this may a few seconds up to a minute to segment the image. 


### Contact

Please post any questions to the [MuscleMap issue page](https://github.com/MuscleMap/MuscleMap/issues).

## Contributing

Contributions to this extensions are welcome. Please send a pull request with any suggested changes. 

[MuscleMap_Slicer3D.png]: MuscleMap_Slicer3D.png