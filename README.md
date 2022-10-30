# <b>S</b>eparate <b>F</b>or <b>M</b>e <b>P</b>lease

Quickly separate images using clustering based on features extracted through a convolutional neural network.

```
$ stmp [FOLDER]
```

### Example

Before
```
.
├── cat.jpg
└── dog.jpg
```

```
$ stmp sample --num-clusters 2
```

After
```
├── cluster_0
│   └── cat.jpg
└── cluster_1
    └── dog.jpg
```


### Dependencies
- Opencv
- [Onnxruntime 1.80](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.0), just download and put it in your `/usr/lib` folder.


Then download (or compile) the `sfmp` binary and place it in your path bin dir. 