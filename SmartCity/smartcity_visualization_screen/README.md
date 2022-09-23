# MatrixOne数据库读取及智慧园区可视化大屏展示

此处展示前端页面使用Javascript读取MatrixOne MySQL数据库的案例。

目录
========

* [MatrixOne安装及启动MySQL服务](#matrixone安装及启动mysql服务)
* [nodejs安装以完成前后端交互](#nodejs安装以完成前后端交互)

## MatrixOne安装及启动MySQL服务
### MatrixOne安装

使用MatrixOne(稳定版)代码构建MatrixOne。

1. 获取使用0.5.1稳定版分支。

```
git clone https://github.com/matrixorigin/matrixone.git
cd matrixone
git checkout 0.5.1
```

2. 编译程序。

```
make config
make build
```

3. 启动MatrixOne服务。

```
./mo-server system_vars_config.toml
```

### 连接到MatrixOne服务

1. 安装MySQL客户端。

此处需安装Oracle MySQL客户端。

2. 连接到MatrixOne服务

```
$ mysql -h IP -P PORT -uUsername -p
```

测试账号为：

* user:dump
* password:111

```
$ mysql -h 127.0.0.1 -P 6001 -udump -p
Enter password:
```

更多使用详情查看：https://github.com/matrixorigin/matrixone

## nodejs安装以完成前后端交互

本案例通过nodejs完成前后端交互。js脚本文件分为client端和server端。client端对应前端页面，通过html页面运行；server部署于服务器端，用于在连接服务器上部署的matrixone MySQL数据库。

### 1. 安装nondejs

此操作在服务器端进行。

```
sudo apt-get install nodejs
sudo apt-get install npm
```

### 2. 服务器端

服务器端代码对应 ./js/connect_matrixone_mysql_server.js

在服务器端新建文件夹放置该代码，进入文件夹。

安装所需模块。

```
npm install express
npm install mysql
```

### 3. 客户端：

客户端代码对应 ./js/connect_matrixone_mysql_client.js

使用时需修改服务器ip地址为部署的服务端地址。