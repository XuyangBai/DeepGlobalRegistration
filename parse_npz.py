import numpy as np

def parse_stats(stats):
    correct_pairs = np.where(stats[:, 0] > 0)
    RR = np.mean(stats[:, 0])
    TE = stats[correct_pairs].mean(0)[1]
    RE = stats[correct_pairs].mean(0)[2]
    precision = stats.mean(0)[5]
    recall = stats.mean(0)[6]
    f1 = 2*precision*recall / (precision + recall)
    time = stats.mean(0)[3]
    print(f"Safe Guard: {np.sum(stats[:, 4])}/{stats.shape[0]}")
    print(f"{stats.shape[0]} Pairs: RR={RR*100:.2f}%, RE={RE:.2f}deg, TE={TE*100:.2f}cm, \
    Precision={precision*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%, time={time:.2f}s")

print("FPFH, indoor:")
data = np.load('kitti-stats_DeepGlobalRegistration_noicp_fpfh_indoor.npz')['stats']
parse_stats(data)

print("FPFH, retrain:")
data = np.load('kitti-stats_DeepGlobalRegistration_noicp_fpfh_retrain.npz')['stats']
parse_stats(data)

print("FCGF, indoor:")
data = np.load('kitti-stats_DeepGlobalRegistration_noicp_fcgf_indoor.npz')['stats']
parse_stats(data)

print("FCGF, retrain:")
data = np.load('kitti-stats_DeepGlobalRegistration_noicp_fcgf_retrain.npz')['stats']
parse_stats(data)


## 3DMatch
print("3DMatch FPFH DGR w/o s.g. :")
data = np.load('3dmatch-stats_DeepGlobalRegistration_noicp_fpfh_nosg.npz')['stats'][0]
parse_stats(data)

print("3DMatch FPFH DGR :")
data = np.load('3dmatch-stats_DeepGlobalRegistration_noicp_fpfh.npz')['stats'][0]
parse_stats(data)
