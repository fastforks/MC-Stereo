# MC-Stereo: speed report (mirror of the upstream code, README only)

> **You are on `benchmark-report`.** This repository is a public research mirror of [MiaoJieF/MC-Stereo](https://github.com/MiaoJieF/MC-Stereo) (base `411222c`; `main` mirrors upstream). `benchmark-report` differs from upstream only in this README block; it contains no code changes.
> **Licence:** upstream has no licence file (all rights reserved), so this fork adds only this report to the upstream code. The code changes behind the numbers below exist only as local patches (to be offered to the authors as pull requests, together with a request for a licence); the fastest row needs custom kernels and is not published. Weights are not included; get them from the [upstream link](https://drive.google.com/drive/folders/1WJiZxZtDnrsN1jIvbASCaLzUwbmf_0NT?usp=drive_link).

**Accuracy caveat:** the released Scene Flow checkpoint is poor zero-shot (ETH3D EPE 4.4 px / bad-1 21.3 %, Middlebury-H EPE 4.3 px / bad-2 14.2 %, all pairs, upstream fp32). Speed and agreement with upstream are therefore the point here, not absolute accuracy; MC-Stereo's multi-peak top-k also makes the output slightly chaotic (even two fp32 runs with different memory layout differ, see below).

MC-Stereo, Scene Flow checkpoint, 32 iterations, batch 1, input padded to /32 as `evaluate_stereo.py`. RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-12, one cold process per row, 3 warm-ups, median of 10; memory = torch peak allocated / reserved over the timed runs.

| level | KITTI ms | Midd-H ms | speed-up at Midd-H | peak MB at Midd-H alloc / reserved | accuracy vs upstream fp32 (Midd-H / ETH3D subset: mean abs diff px) | EPE vs GT (ETH3D / Midd-H subset) |
|---|---|---|---|---|---|---|
| upstream default (fp32) | 543 | 1808 | 1.00x | 2205 / 2692 | reference | 4.50 / 1.15 |
| upstream + recommended fast flags: `mixed_precision=True` (its own switch) + channels-last weights/inputs | 334 | 1111 | 1.63x | 1887 / 2106 | 0.10 / 0.42 | 4.56 / 1.14 |
| + realistic upstream PRs (4 small patches, with the flags above) | 287 | 999 | 1.81x | 1923 / 2182 | 0.09 / 0.42 | 4.61 / 1.15 |
| local portable fixes (plain PyTorch; not published) | 287 | 997 | 1.81x | 1923 / 2182 | 0.09 / 0.42 | 4.61 / 1.15 |
| local CUDA kernels (not published) | 130 | 459 | 3.94x | 1033 / 1384 | 0.09 / 0.49 | 4.69 / 1.15 |

**Realistically mergeable upstream: 1.81x of 3.94x** (row "+ realistic PRs", incl. flags; max row = 3.94x). All plain-PyTorch changes are PR-sized, so the local portable row equals the PR row; there is no separate published code branch.
Startup (model ready from process launch) 3.3 s for all rows; first call at a new size costs +0.5 to +0.6 s over steady state at the first size (cuDNN setup) and ~0 s for later sizes. `cudnn.benchmark` is not used: no speed-up, +1.3 s to +2.4 s at every new size.
Accuracy columns: ETH3D every 3rd pair (9 of 27, bad-1 vs GT 19.4 to 20.2 % for all rows) and Middlebury-H every 3rd pair (5 of 15). The difference to upstream fp32 is at the level of the precision noise: fp32 with channels-last alone already differs by 0.05 px (Midd-H) / 0.33 px (ETH3D) because of the top-k peak selection. KITTI is used for latency only.

Realistic PRs (each alone on upstream `main`, with the flags row as its base; ms saved vs the flags row at KITTI / Midd-H; drafted, not posted):
- `pr/gather-lookup`: cost-volume lookup by gather instead of `grid_sample` + `torch.unique` host sync (same values; also lets `torch.export` trace it): -20 / -0 ms.
- `pr/merged-zr-gru`: z and r gates of each ConvGRU in one conv: -12 / -44 ms.
- `pr/skip-unused-mask`: upsampling mask only for the last iteration in test mode: -17 / -68 ms.
- `pr/batched-fnet`: feature extractor once on left+right as a batch of 2: +1 / +2 ms on this GPU (kept: fewer launches, no loss; can be dropped).
The four together: -47 / -112 ms (not additive; they overlap on launch overhead). In plain fp32 (no flags) the four give 1808 -> 1616 ms at Midd-H (1.12x).

Fork-only: the max row (fused fp16 NHWC convolutions with CUTLASS GRU epilogues, prebuilt kernels; a per-GPU build/binary, and the code could not be shared without a licence).

Source and raw data: `results/2026-10-12_mc-stereo_release/` in the local research results (harness `mc_bench.py`, `bench.jsonl`, `accuracy.json`).

---

# MC-Stereo (3DV 2024)

This repository contains the source code for our paper:

[MC-Stereo: Multi-Peak Lookup and Cascade Search Range for Stereo Matching](https://arxiv.org/pdf/2311.02340.pdf)<br/>
Miaojie Feng, Junda Cheng, Hao jia, Longliang Liu, Gangwei Xu, Xin Yang<br/>

<img src="./images/MC-Stereo.png">

<img src="./images/vis_kitti.png">

## Comparision

| Method          | KITTI-2012<br>(2-noc) | KITTI-2012<br>(2-noc-ref) | KITTI-2015<br>(D1-all) |
|:--------------- |:---------------------:|:-------------------------:|:----------------------:|
| ACVNet          | 1.83                  | 11.42                     | 1.65                   |
| RAFT-Stereo     | -                     | -                         | 1.91                   |
| IGEV-Stereo     | 1.71                  | 7.29                      | 1.59                   |
| CREStereo       | 1.72                  | 9.71                      | 1.69                   |
| MC-Stereo(Ours) | 1.55                  | 6.82                      | 1.55                   |

## Requirements

```
conda create -n MC_Stereo python=3.7
conda activate MC_Stereo
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install tqdm
pip install timm==0.5.4
```

## Required Data

To evaluate/train MC-Stereo, you will need to download the required datasets.

* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [CREStereo](https://github.com/megvii-research/CREStereo/tree/master)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)

## Evaluation

Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1WJiZxZtDnrsN1jIvbASCaLzUwbmf_0NT?usp=drive_link).

To evaluate on Scene Flow, run

```Shell
sh evaluate.sh
```

## Training

To train on Scene Flow, run

```Shell
sh train_sceneflow.sh
```

To train on KITTI, run

```Shell
sh train_kitti.sh
```

To train on ETH3D, run

```Shell
sh train_eth3d.sh
```

## Submission

For submission to the KITTI benchmark, run

```Shell
python save_disp.py
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{feng2023mc,
  title={MC-Stereo: Multi-peak Lookup and Cascade Search Range for Stereo Matching},
  author={Feng, Miaojie and Cheng, Junda and Jia, Hao and Liu, Longliang and Xu, Gangwei and Yang, Xin},
  journal={arXiv preprint arXiv:2311.02340},
  year={2023}
}
```

## Contact

Please feel free to contact me (Miaojie) at fmj@hust.edu.cn.

## Acknowledgements

This project is heavily based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), We thank the original authors for their excellent work.
