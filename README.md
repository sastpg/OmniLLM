# OmniLLM

<div align="center">
    <a href="https://github.com/sastpg/OmniLLM/tree/main">
        <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python Version"/>
    </a>
    <a href="https://github.com/sastpg/OmniLLM/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
    </a>
</div>

⚡ Build open-source large language model(LLM) service ⚡

Read this in [English](README_en.md)

![](./images/web_demo.png)

## 项目更新

- ``2024/07/14``: 支持模型流式输出。
- 🔥 ``2024/07/08``: 实现工具调用（也叫函数调用）的 API 支持，更多调用细节请前往 [查看]()。
- 🔥 ``2024/06/17``: 支持 LLaMA3, Qwen2, GLM4 以及它们微调后模型的 API 调用，发布在网站中使用的简单示例。

## 快速开始
### Python 环境
推荐您使用 Conda 进行 python 包管理，关于如何安装和使用 Conda 请见[Conda 官方文档](https://conda.io/en/latest/index.html)。

1. 克隆本仓库并进入项目文件夹
```bash
git clone https://github.com/sastpg/OmniLLM.git
cd OmniLLM
```

2. 安装 Python 包
```bash
conda create -n omnillm python=3.12
conda activate omnillm
cd service  # 模型 API 服务
pip install -r requirements.txt
cd ../web  # 以下命令可不执行，如果您只需要 API 服务而不需要网页 Demo
pip install -r requirements.txt
```

### 模型下载
