# MatrixOne SmartCity Showcase

This project takes MatrixOne open source distributed HSTAP database and open source streaming distributed storage Pravega as the core to build an end-to-end smart city digital system, including video acquisition, transmission, model reasoning, data cleaning, storage, analysis, visualization and other functional modules. On the basis of system implementation, BLOB data types are implemented for MatrixOne to enrich the data types supported by MatrixOne.

It mainly includes the following functional modules:

+ camera

+ persistent stream storage Pravega

   + Pravega introduction

   Rebuild new storage for streaming computing

   Pravega provides a new storage abstraction stream for continuous and unbounded data. It is a persistent, elastic, append only and unlimited byte sequence with good performance and strong consistency.

   Using Pravega, you can capture real-time data streams for analysis. Today, real-time streaming is everywhere. Data from mobile phones, social media streams, videos, sensors, drones, etc. is continuous and unlimited ("real-time streaming").

   Pravega's elastic architecture can be applied to various scenarios. It can process a large amount of real-time data, which is helpful to capture data from spacecraft launch; It can be used to promote more effective traffic flow in smart cities, or compare large commercial construction projects with design renderings to ensure their punctuality and accuracy.

   This is just the beginning. Pravega can be used in any industry and any use case involving real-time streaming data.

   + significance of Pravega in the project

      + pravega played the core role of data pipeline in the artificial intelligence scene at the edge of the whole intelligent industrial park, and completed the collection and reversal of end-to-end data. At the same time, it can isolate the data at all levels through the abstraction of stream.

      + in the process of video data collection, the edge video terminal is connected, and multi-channel video digital can be quickly injected into pravega's stream at the same time through GStreamer.

      + in the process of video processing, recognition and analysis, the edge AI engine consumes video data in real time through pravega's data access interface. Using pravega's characteristics of low delay and high throughput, the edge AI engine can realize real-time prediction, and write the AI result data into pravega's stream to provide materials for downstream tasks.

      + for data application and visualization, use JDBC to synchronize AI result data to matrixone to complete the transmission of the last value link of data. Matrixone can provide complex data organization and integration capabilities and display them in a visual way.

+ video reasoning module

+ high performance database MatrixOne

+ visual WebUI

The overall architecture is shown in the figure below:
