# Grouping Neural Network

### main.py
Harry Potter's Grouping Hat Neural Net
This is Harry Potter's grouping hat using convolutional neural network.

### main_cap.py
Dynamically Grouping the face images acquired from the camera with the learned model.

## Requirements

### main.py
python27 @2.7.12_1, chainer @1.20.0.1_0, py27-numpy @1.11.1_0+gfortran and py27-matplotlib @1.5.1_1+cairo+tkinter

### main_cap.py
python27 @2.7.12_1, chainer @1.20.0.1_0, py27-numpy @1.11.1_0+gfortran and opencv @3.2.0_1+contrib+java+python27+qt4+vtk

## Install

```git clone```to your computer from here.

## Usage

### main.py
To run it;
```
python main.py [-h] [--indir [INDIR]] [--outdir [OUTDIR]] [--model [MODEL]]
               [--gpu GPU] [--imsize IMSIZE] [--kcross KCROSS] [--epoch EPOCH]
               [--batchsize BATCHSIZE] [--phase PHASE] [--method METHOD]
```

### main_cap.py
To run it;
```
python main_cap.py [-h] [--model [MODEL]] [--gpu GPU] [--frame FRAME] [--method METHOD]
```
or
```
./exe_cap.sh
```
exe_cap.sh : Automatically download the learned model and then execute main_cap.py.

## Author
tokkuman
