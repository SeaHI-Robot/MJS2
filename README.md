<h1 align="center">
    MJS2
</h1>

<p align="center">
<i> <b>"MuJoCo Sim2Sim"</b> </i>
</p>

<p align="center">
    <img
        alt="MuJoco"
        src="https://img.shields.io/badge/MJC-Python--Binding-a6e3a1?style=for-the-badge&colorA=363A4F&logo=python&logoColor=D9E0EE">
    <img
        alt="Pytorch"
        src="https://img.shields.io/badge/Pytorch-Jit-fab387?style=for-the-badge&colorA=363A4F&logo=pytorch&logoColor=D9E0EE">
    <img
        alt="Code Size"
        src="https://img.shields.io/github/languages/code-size/SeaHI-Robot/MJS2?colorA=363A4F&colorB=b4befe&logo=gitlfs&logoColor=D9E0EE&style=for-the-badge">
</p>
<br>

## 🪷 Introduction

- This repo hosts a sim2sim template code for validating RL locomotion policy in [MuJoCo](https://mujoco.readthedocs.io/en/latest/overview.html).
- RL policies are loaded as torch jit `.pt` file by default.
- Editting `.yaml` files in cfg directory is always your first choice during sim2sim testing.
- Integrated with a keyboard command control functionality through `pynput`.

## 📁 Structure
```
├── cfg
│   ├── TRON1.yaml
│   └── A1.yaml
├── policy
│   ├── TRON1
│   │   └ ...
│   └── A1
│       └ ...
├── resources
│   ├── TRON1
│   └── A1
└── scripts
    ├── TRON1
    │   ├── sim2sim.py
    │   └── utils.py
    └── A1
        ├── sim2sim.py
        └── utils.py
```

## 🤖 Adding New Robot
1. Create a folder `<your_robot_name>` in `./resources/<your_robot_name>/` containing your robot mjcf description files.
2. Put your policy files into `./policy/<your_robot_name>/`
3. Implement a new `./cfg/<your_robot_name>.yaml` file.
4. Implement`./scripts/<your_robot_name>/sim2sim.py` and `./scripts/<your_robot_name>/utils.py` according to your demands.

<br>
