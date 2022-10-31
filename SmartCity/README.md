# SmartCity Showcase

This project builds an end-to-end smart city digital system based on open-source distributed HSTAP database MatrixOne and open-source streaming distributed storage Pravega.

Contents
========


- [SmartCity Showcase](#smartcity-showcase)
- [Contents](#contents)
  - [System Architecture](#system-architecture)
  - [Components](#components)
  - [Requirements](#requirements)
    - [MatrixOne](#matrixone)
      - [Key Features](#key-features)
    - [Pravega](#pravega)
  - [Getting start](#getting-start)
    - [Build Matrixone](#build-matrixone)
      - [Step 1. Install Go (version 1.19+ is required).](#step-1-install-go-version-119-is-required)
      - [Step 2. Checkout the source code to build MatrixOne](#step-2-checkout-the-source-code-to-build-matrixone)
    - [Build Pravega](#build-pravega)
      - [Step 1. Install Java (version 11+ is required).](#step-1-install-java-version-11-is-required)
      - [Step 2. Checkout the source code](#step-2-checkout-the-source-code)
    - [GStreamer Plugins for Pravega](#gstreamer-plugins-for-pravega)
      - [Step 1. Build Gstreamer](#step-1-build-gstreamer)
      - [Step 2. Install Rust](#step-2-install-rust)
      - [Step 3. Build and Install GStreamer Plugins for Pravega](#step-3-build-and-install-gstreamer-plugins-for-pravega)
    - [Gst-Python](#gst-python)
      - [Step 1. Install python lib](#step-1-install-python-lib)
      - [Step 2. Build gst-python from source](#step-2-build-gst-python-from-source)
        - [Troubleshooting](#troubleshooting)
    - [Model setup](#model-setup)
      - [Install requirement](#install-requirement)
      - [Install yolox](#install-yolox)
  - [Example workflow](#example-workflow)
    - [Launch](#launch)
    - [Plugins Checkout](#plugins-checkout)
    - [Data Inject](#data-inject)
    - [Model Detection](#model-detection)
    - [Matrixone Analysis](#matrixone-analysis)
    - [Frontend Display](#frontend-display)
  - [Contributors](#contributors)



## System Architecture
![architecture](/SmartCity/images/architecture.jpg)

## Components 

Followings are the fundamental components in our system:
- camera
- persistent stream storage Pravega
- video reasoning module
- high performance database MatrixOne
- visual WebUI

## Requirements
### MatrixOne
[MatrixOne](https://github.com/matrixorigin/matrixone) is a future-oriented hyper-converged cloud & edge native DBMS that supports transactional, analytical, and streaming workloads with a simplified and distributed database engine, across multiple data centers, clouds, edges and other heterogeneous infrastructures.
#### Key Features
- Hyperconverged Engine: MatrixOne supports hybrid workloads by a monolithic database engine: transactional, analytical, time-series, machine learning, - etc. MatrixOne also supports in-database streaming processing with a built-in streaming engine.
- Cloud & Edge Native: MatrixOne is real infrastructure agnostic, it supports seamless workload migration and bursting among different locations and infrastructures, as well as multi-site active-active with industry-leading latency control.
- Extreme Performance: MatrixOne supports blazing-fast queries on vast datasets while guaranteeing strong consistency and high scalability.
### Pravega
[Pravega](https://github.com/pravega/pravega) provides a new storage abstraction stream for continuous and unbounded data. It is a persistent, elastic, append only and unlimited byte sequence with good performance and strong consistency.

   Using Pravega, you can capture real-time data streams for analysis. Today, real-time streaming is everywhere. Data from mobile phones, social media streams, videos, sensors, drones, etc. is continuous and unlimited ("real-time streaming").

   Pravega's elastic architecture can be applied to various scenarios. It can process a large amount of real-time data, which is helpful to capture data from spacecraft launch; It can be used to promote more effective traffic flow in smart cities, or compare large commercial construction projects with design renderings to ensure their punctuality and accuracy.

   This is just the beginning. Pravega can be used in any industry and any use case involving real-time streaming data.

   + significance of Pravega in the project

      + pravega played the core role of data pipeline in the artificial intelligence scene at the edge of the whole intelligent industrial park, and completed the collection and reversal of end-to-end data. At the same time, it can isolate the data at all levels through the abstraction of stream.

      + in the process of video data collection, the edge video terminal is connected, and multi-channel video digital can be quickly injected into pravega's stream at the same time through GStreamer.

      + in the process of video processing, recognition and analysis, the edge AI engine consumes video data in real time through pravega's data access interface. Using pravega's characteristics of low delay and high throughput, the edge AI engine can realize real-time prediction, and write the AI result data into pravega's stream to provide materials for downstream tasks.

      + for data application and visualization, use JDBC to synchronize AI result data to matrixone to complete the transmission of the last value link of data. Matrixone can provide complex data organization and integration capabilities and display them in a visual way.


## Getting start
This section assumes that you are using Ubuntu version 20.04+. Some libraries and programs may not be compatible with older versions. Docker is recommended.

Clone this code repository first.
```bash
git clone https://github.com/matrixorigin/matrixone-demo.git
```
### Build Matrixone
Building Matrixone by following steps. You can also build Matrixone refer to [official document](https://github.com/matrixorigin/matrixone/blob/main/README.md).
#### Step 1. Install Go (version 1.19+ is required).
```bash
wget https://golang.google.cn/dl/go1.19.1.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.19.1.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```
#### Step 2. Checkout the source code to build MatrixOne
Get the MatrixOne(Master branch) code and build.
```bash
git clone https://github.com/matrixorigin/matrixone.git
cd matrixone
make build
```
Launch MatrixOne server:
```bash
./mo-service -cfg ./etc/cn-standalone-test.toml
```
### Build Pravega
Building Pravega by following steps. You can also build Pravega refer to [official document](https://github.com/pravega/pravega/blob/master/README.md).
#### Step 1. Install Java (version 11+ is required).
```bash
sudo apt-get install openjdk-11-jdk
sudo update-alternatives --config java
```
#### Step 2. Checkout the source code
```bash
git clone https://github.com/pravega/pravega.git
cd pravega
```
Build and run standalone mode:
```bash
./gradlew startStandalone
```

### GStreamer Plugins for Pravega
GStreamer is a pipeline-based multimedia framework that links together a wide variety of media processing systems to complete complex workflows. In this demo, GStreamer is used to support Pravega as pipeline to deliver video stream for fianl model detection.
Building GSt-Pravega by following steps. You can also build Pravega refer to [official document](https://github.com/pravega/gstreamer-pravega#getting-started).
#### Step 1. Build Gstreamer
```bash
sudo apt-get install \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libatk1.0-dev \
    libcairo-dev \
    libges-1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstrtspserver-1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libpango1.0-dev \
    libssl-dev
```
#### Step 2. Install Rust
```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
rustup update
```
Update environment variables.
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```
#### Step 3. Build and Install GStreamer Plugins for Pravega
Use Cargo to build the GStreamer Plugins for Pravega from source.
```bash
git clone https://github.com/pravega/gstreamer-pravega.git
cd gstreamer-pravega
cargo build --package gst-plugin-pravega --locked --release
```
Install the plugin by moving to gstreamer plugins folder.
```bash
sudo cp target/release/*.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
```

### Gst-Python
In the early version of Gstraemer(maybe 1.20-), we need to manually generate `libgstpython.so` to bind Gstreamer with Python code. If your Gstreamer plugins already have it, you can skip to next part.
#### Step 1. Install python lib
Need Python 3.7+.
```bash
sudo apt install python3-gi python3-gi-cairo python-gi-dev
pip install pygobject
```
#### Step 2. Build gst-python from source
Find the [src](https://gstreamer.freedesktop.org/src/gst-python/) match with your Gstreamer version. Then get and build it.
```bash
wget https://gstreamer.freedesktop.org/src/gst-python/gst-python-1.16.3.tar.xz
tar -xvf gst-python-1.16.3.tar.xz
git checkout $BRANCH
./autogen.sh --disable-gtk-doc --noconfigure
# --with-libpython-dir = location of libpython*.so
./configure --with-libpython-dir="/usr/lib/x86_64-linux-gnu"
make
sudo make install
```
Install the plugin by moving it to gstreamer plugins folder.
```bash
sudo cp /usr/local/gstreamer-1.0/*.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
```
**Update environment variables. PYTHONPATH is somewhere include newly generate `gi/override files`. GST_PLUGIN_PATH is the former Gstreamer plugins folder path.**
```bash
export PYTHONPATH=/usr/local/lib/python3.8/site-packages
export GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/
```
verify the plugin availability.
```bash
cd matrixone-demo/SmartCity
GST_PLUGIN_PATH=$PWD/plugins:$GST_PLUGIN_PATH \
gst-inspect-1.0 example_transform_cv
```
You should see some plugin details.

##### Troubleshooting
1. Remove gstreamer's cache to reload plugins and display error messages when there is a problem.
    ```bash
    rm -rf ~/.cache/gstreamer-1.0/
    ```
2. Check GST_PLUGIN_PATH export as described above if you are receiving`No such element or plugin 'myplugin'`.
3. Check PYTHONPATH export as described above if you are receiving`gst is no dict'`.

### Model setup
#### Install requirement
```bash
pip install --upgrade --requirement requirements.txt
```
#### Install yolox
```bash
cd SmartCity/models/face/YOLOX
python setup.py develop
```
## Example workflow
This demo take standalone mode as example. You can also extend it to multiple servers.
### Launch
Lanch Matrixone, Pravega first.
```bash
cd matrixone/
./mo-service -launch etc/launch-tae-logservice/launch.toml
```
```bash
cd pravega/
./gradlew startStandalone
```
Create the Matrixone database scheme.
```sql
create database park;
use park;
```
> if you want to clear all the data that Matrixone stored, just delete the folder `store`.
> 
### Plugins Checkout
```bash
cd SmartCity/
GST_PLUGIN_PATH=$PWD/plugins:$GST_PLUGIN_PATH \
gst-inspect-1.0 example_transform_cv
```
### Data Inject
```bash
python video_file_to_pravega.py \
--pravega-scope examples --pravega-stream mystream1 \
--source-uri path-to-mp4-video-file
```
### Model Detection
Modify the plugin file `plugins/python/example_transform_cv.py`.
In line 25:
```python
model_dir = "the path to your 'models' folder"
```
Then run the model detection.
```bash
python pravega_video_to_cv.py
```
The program continuously detects incoming streaming video.
### Matrixone Analysis
You can use the MySQL command-line client to connect to MatrixOne server. The connection string is the same format as MySQL accepts. You need to provide a user name and a password.  
Use the built-in test account for example.
```bash
mysql -h 127.0.0.1 -P 6001 -udump -p111
```
Then analysis data with sql sentence.
### Frontend Display
A visualization case looks like this.

![](./smartcity_visualization_screen/images/screen_example1.jpg)

See README in `smartcity_visualization_screen` folder for more details.

## Contributors
<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/sukki37">
            <img src="https://avatars.githubusercontent.com/u/77312370?v=4" width="30;" alt="sukki37"/>
            <br />
            <sub><b>Maomao</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/yjw1268">
            <img src="https://avatars.githubusercontent.com/u/29796528?v=4" width="30;" alt="yjw1268"/>
            <br />
            <sub><b>Ryan</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/BUPT-NingXinyu">
            <img src="https://avatars.githubusercontent.com/u/44694099?v=4" width="30;" alt="BUPT-NingXinyu"/>
            <br />
            <sub><b>Xinyu Ning</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

