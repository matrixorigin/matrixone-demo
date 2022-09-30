# MatrixOne数据库读取及智慧园区可视化大屏展示

此处展示前端页面使用Javascript读取MatrixOne MySQL数据库的案例。

本前端采用HTML+js+CSS搭建页面。图表模块采用echarts组件进行构建。

对于页面尺寸设计，本案例前端页面的定位为智慧园区大屏，因此前端应适应横屏输出。固定页面最小宽度1024px最大宽度1920px，使用基于flexible.js + rem (相对长度) 的方式使得页面对不同大小的屏幕进行适配。

对于功能模块设计。前端页面包含7个功能模块：设备数量监测模块、车辆监测模块、性别年龄模块、人流量监控模块、告警模块、工服安全帽佩戴模块、口罩佩戴模块。


目录
========

* [页面展示及功能介绍](#页面展示及功能介绍)
* [MatrixOne安装及启动MySQL服务](#matrixone安装及启动mysql服务)
* [nodejs安装以完成前后端交互](#nodejs安装以完成前后端交互)

## 页面展示及功能介绍
![](./images/screen_example1.jpg)
图1 页面展示

![](./images/screen_example2.jpg)
图2 回传图片展示

①	设备数量监测模块

显示园区中设备总数、运行设备数及异常设备数。目前为静态模块。

②	车辆检测模块

显示园区中登记车辆数及现有车辆数。采用echart组件中的柱状图进行自定义配置以显示园区中剩余车位容量。目前为静态模块。

③	性别年龄监测模块

显示对人脸检测时记录下的年龄和性别结果。使用echart组件中的饼形图分别显示人员年龄和性别比例。

后端交互方式：查询人脸信息表中不同年龄段及不同性别条目数量并返回。

④	人流量监控模块

显示今日园区人流量及当前园区内剩余人数。下方使用echart显示园区随时间而变化的人流量折线图，其中蓝色线条表示入场人流量，绿色线条显示出场人流量。

⑤	告警模块

显示对烟火和异常井盖的告警。正常状态下为绿色字体，文字内容为“正常“；异常状态下为红色字体，文字内容为”告警“。点击”告警“二字，在屏幕中间显示异常处摄像头存储的图片。

后端交互方式：对于烟火告警，查询烟火信息表中按时间排列的最后一条数据返回给前端页面，读取该数据中environment字段存储的字符串信息，对字符串做截断读取烟火判定的置信度。如果置信度大于0.5则显示告警；对于井盖告警，按时间排序查询井盖检测信息表，如果最近的告警数据时间小于2000秒，则在前端显示告警。

⑥	工服安全帽佩戴模块

显示摔倒检测告警，该部分与告警模块运行逻辑相同。使用echart柱状图显示工服安全帽佩戴比例。
后端交互方式：查询摔倒检测表，按照时间排序返回最后一条数据，如果该数据时间与当前时间间隔小于2000秒则显示告警。工服安全帽检测为静态模块。

⑦	口罩佩戴模块

使用echart南丁格尔玫瑰图显示口罩佩戴情况。

后端交互方式：查询人脸检测信息表，返回字段口罩佩戴情况分别为0和1的条目数量，将数量传入echart组件相应参数进行显示。

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

(开发版本使用)
```
./mo-service -cfg ./etc/cn-standalone-test.toml
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