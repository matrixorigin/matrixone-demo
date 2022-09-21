# MatrixOne数据库读取及智慧园区可视化大屏展示

此处展示前端页面使用Javascript读取MatrixOne MySQL数据库的案例。

目录
========

* [MatrixOne安装及启动MySQL服务](#matrixone安装及启动mysql服务)

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