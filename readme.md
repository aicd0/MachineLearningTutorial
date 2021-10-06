1. 运行 _03_similarity_measurement.py
2. 运行 _04_feature_extraction.py
3. 在自动生成的 outputs 目录下查看结果

## 说明
* 相似性度量指标采用曼哈顿距离(Manhattan Distance)，窗口大小为500，每次移动50个单位距离，两个窗口不允许重叠。
* 特征提取所选择的的指标分别为：偏度(Skewness)、峰度(Kurtosis)、Renyi熵 __[1]__ (Renyi Entropy)、近似熵 __[2]__ (Approximate Entropy)、样本熵 __[3]__ (Sample Entropy)和排列熵 __[4]__ (Permutation Entropy)。

__[1]__ alpha=2  
__[2]__ m=6, r=0.15  
__[3]__ m=6, r=0.15  
__[4]__ m=6, step=1