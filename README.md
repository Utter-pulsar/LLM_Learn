# LLM_Learn
#### Utter_pulsar

----

> 这是一个基于```modelscope```、```langchain```以及```huggingface```基础上学习大语言模型的学习笔记

----
## 安装

- 环境安装

请按照[modelscope](https://modelscope.cn/docs/intro/quickstart)、[langchain](https://python.langchain.com/docs/tutorials/)
的文档安装环境，模型文件基本都是从[huggingface](https://huggingface.co/models)上面下载的，不过这个网站需要翻墙。

- 虚拟环境

安装必要的虚拟环境请使用管理员权限打开```CMD```：

```CMD
git clone 
cd LLM_Learn
conda create -n LLM python=3.10
conda activate LLM
pip install -r requirements.txt
```
----
## 项目结构解释

本项目总共有五个文件夹：

- FineTuneDataset: 这里是存放用来微调的数据集的
- FineTuneModels: 这里是存放微调过的模型的
- Lib: 这是尝试将langchain与modelscope合并的一个包
- RAGDataset: 这里存放的是RAG的知识库的地方
- models: 这里是存放所有本地化模型文件的地方，文件下载之后请将需要的模型文件压缩包直接解压到```./models/```目录下面。地址是地址是[百度网盘/models](https://pan.baidu.com/s/1OAiq4ns-VJbu1D9bLlo_YA ) ,提取码: iwer

```.py```文件均带有不同```[]```开头的文件名。其中：

- ```[First]```表示初次尝试，可以运行但是代码比较乱
- ```[Formal]```表示已经整理好的代码
- ```[Half]```表示可以运行但是结果不对
- ```[Intro]```表示这是一个小功能但是非常简单，也没有整理
- ```[Simple]```表示这是一个尝试，虽然能运行，但是需要再看看细节

所有的```.py```文件均可单独直接运行。
