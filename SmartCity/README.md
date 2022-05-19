# SmartCity Showcase

This project builds an end-to-end smart city digital system based on open-source distributed HSTAP database MatrixOne and open-source streaming distributed storage Pravega.

Contents
========

* [System Architecture](#system-architecture)
* [Components](#components)
* [Requirements](#requirements)
* [Contrubutors](#contributors)


## System Architecture
![architecture](/SmartCity/images/architecture.jpg)

## Components 

Followings are the fundamental components in our system:

+ camera

+ persistent stream storage Pravega

+ video reasoning module

+ high performance database MatrixOne

+ visual WebUI

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
        <a href="https://github.com/BUPT-NingXinyu">
            <img src="https://avatars.githubusercontent.com/u/44694099?v=4" width="30;" alt="BUPT-NingXinyu"/>
            <br />
            <sub><b>Xinyu Ning</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

