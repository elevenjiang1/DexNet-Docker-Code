# DexNet

> Detail environment setup will add later if it is needed.

Owing that origin dex-net code is too old and some code is difficult to run, here provide a Docker and example code for DexNet.

- Docker:

  https://hub.docker.com/repository/docker/elevenjiang/dexnet

- Code:

  https://github.com/elevenjiang1/DexNet-Docker-Code



## 1. Usage

1. Use scripts in dexnet_docker to create a container

   **!!!Do remember to change your own code path in create_gl_container.sh!!!**

2. A serial of make dataset example code are in dex-net/make_dataset_code, they are base on dex-net/tools and many other issues in dex-net





feel free to contact me with elevenjiang8@gmail.com



******

由于原始DexNet的环境配置太过麻烦,为此使用了一个Docker进行配置

- Docker:

  https://hub.docker.com/repository/docker/elevenjiang/dexnet

- Code:

  https://github.com/elevenjiang1/DexNet-Docker-Code



此环境基于Ubuntu_gl_16.04CUDA进行配置,单独的Ubuntu_CUDA可能会导致最终现实出问题



## 1. 使用方式

> 需要一定的前置知识,如Docker的基本使用,文件挂载等

1. 使用dexnet_docker中的两个脚本文件进行容器的创建和执行

   **!!!需要对create_gl_container.sh中的路径进行修改!!!**

2. 在dex-net/make_dataset_code中包含有一系列进行数据集制作的代码,基于本身tools中的代码进行实现,如果有数据制作的需要可以进行参考





## 2. 其他参考

> 配置过程中还有自己配置的文件说明,可以参考Docs/6.DexNet部署.md





可以通过邮件 elevenjiang8@gmail.com 或  1206413225@qq.com 联系我



*****

All code are reference base on the origin dex-net and gqcnn code

所有的代码基于原始的dex-net和gqcnn代码

- https://github.com/BerkeleyAutomation/dex-net



