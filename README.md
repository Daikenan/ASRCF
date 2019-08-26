# ASRCF
- Visual Tracking via Adaptive Spatially-Regularized Correlation Filters(**CVPR2019 Oral**).

<div align="center">
  <img src="https://github.com/Daikenan/ASRCF/blob/master/faceocc1.gif" width="500px" />
</div>
##　abstract
In this work, we propose a novel adaptive spatially-regularized correlation filters (ASRCF) model to simultaneously optimize the filter coefficients and the spatial regularization weight. First, this adaptive spatial regularization scheme could learn an effective spatial weight for a specific object and its appearance variations, and therefore result in more reliable filter coefficients during the tracking process. Second, our ASRCF model can be effectively optimized based on the alternating direction method of multipliers, where each subproblem has the closed-from solution. Third, our tracker applies two kinds of CF models to estimate the location and scale respectively. The location CF model exploits ensembles of shallow and deep fea-
tures to determine the optimal position accurately. The scale CF model works on multi-scale shallow features to estimate the optimal scale efficiently. Extensive experiments on five recent benchmarks show that our tracker performs favorably against many state-of-the-art algorithms, with real-time performance of 28fps.
## Paper link
- [Google Drive](https://drive.google.com/file/d/1zsUnEmXTLwXqTKytpv3dWTqEreK90_bI/view?usp=sharing)
## Citation
Please cite the above publication if you use the code or compare with the ASRCF tracker in your work. Bibtex entry:
```
@InProceedings{Dai_2019_CVPR,  
  author = {Dai, Kenan and Wang, Dong and Lu, Huchuan and Sun, Chong and Li, Jianhua},  
  title = {Visual Tracking via Adaptive Spatially-Regularized Correlation Filters},  	
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  	
  month = {June},  
  year = {2019}  
}  
```

## Installation
### System: Test on Ubuntu 16.04 and Ubuntu 14.04
It may have some problems in windows 10, but someone has successfully run through, if you encounter some difficult problems, you can submit issues.
### Platfrom: Matlab 2017a(or lower version)
If you want to run it with matlab2018, you may need to add ‘-R2018a’ to the set of args in the function mex_link of vl_compilenn.m e.g.
```
args = horzcat(‘-R2018a’…..
```
1. Clone the GIT repository:
```
 $ git clone https://github.com/Daikenan/ASRCF.git
```
2. Clone the submodules.  
   In the repository directory, run the commands:
```
   $ git submodule init  
   $ git submodule update
```
3. Start Matlab and navigate to the repository.  
   Run the install script:
```
   |>> install
```
4. Run the demo script to test the tracker:
```
   |>> demo_ASRCF
```   
## Use different GPU cards
- We use GPU card 1 by default, if you want to use other GPU cards, such as card 2, you can run these code in matlab Command Window.
```
　opts.gpus=[2];
　prepareGPUs2(opts,ture);
```
## Spatial regularization
- In our demo, we show the spatial adaptive regularization by default, but this is time consuming.
if you want to close it, you need just set `params.show_regularization = 0` in `run_ASRCF.m`.

## Results
- [OTB100](https://drive.google.com/open?id=1pceA0p4C3DvfVK2-U7asPuSh9b-CJ3Hh)
- [TC128](https://drive.google.com/open?id=1ps1BKdxidSsdbck6ynWwozKoKffv9rUA)
- [LaSOT](https://drive.google.com/open?id=1R3RnRJp3vMznuXGQaHiZaZVG_pB9O65e)  

## Contact
- If you have any other question, you can contact me by email: dkn2014@mail.dlut.edu.cn
