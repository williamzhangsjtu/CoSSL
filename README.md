# CoSSL:
This project is the source codes for CoSSL.


The main script for the project is [code/main.py](code/main.py), you can run it by: 
```python
 python pretrain.py [-c config_file]
```
And config_file can be the configuration file [config/config.yaml](config/config.yaml), or you can use other configuration. The training data is not included in the project, you can use your own data and remember to modify 'audio_h5' and 'ref_h5' in [config/config.yaml](config/config.yaml). The input feature file should be in *hdf5* type, and it should contain **key: value** data. Each sample in 'audio_h5' should have the same length, and there should be a corresponding reference in 'ref_h5' for each sample in 'audio_h5'.
