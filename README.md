# <b>S</b>eparate <b>F</b>or <b>M</b>e <b>P</b>lease

Quickly separate images using clustering based on features extracted through a convolutional neural network.

```
sfmp 0.1.0
Separate For Me Please

USAGE:
    sfmp [OPTIONS] <PATH>

ARGS:
    <PATH>

OPTIONS:
    -f, --features <FEATURES>                    [default: 512]
    -h, --help                                   Print help information
    -i, --input-size <INPUT_SIZE>                [default: 224]
    -m, --max-n-iterations <MAX_N_ITERATIONS>    [default: 300]
    -m, --move-images
    -n, --num-clusters <NUM_CLUSTERS>            [default: 2]
    -t, --tolerance <TOLERANCE>                  [default: 0.0001]
    -V, --version                                Print version information
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