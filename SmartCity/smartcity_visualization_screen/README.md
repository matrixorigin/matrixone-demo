# MatrixOne Database Reading and Visual Display

<a href="https://github.com/BUPT-NingXinyu/matrixone-samples/tree/main/SmartCity/smartcity_visualization_screen/README.md">
  <b>English</b>
</a>
  <b>||</b>
<a href="https://github.com/BUPT-NingXinyu/matrixone-samples/tree/main/SmartCity/smartcity_visualization_screen/README_CN.md">
  <b>简体中文</b>
</a>

Here is a case where the front-end page uses Javascript to read the MatrixOne MySQL database.

This case uses HTML+js+CSS to build pages. The chart module is built with [echarts](https://echarts.apache.org/zh/index.html).

For page size design, the front page of this case is positioned as a large screen in the smart park, so the page should adapt to horizontal screen output. The minimum width of a fixed page is 1024px and the maximum width is 1920px The method of js+rem (relative length) enables pages to adapt to screens of different sizes.

For functional module design. The front page includes 7 functional modules: 
* equipment quantity monitoring module
* vehicle monitoring module
* gender and age module
* pedestrian flow monitoring module
* alarm module
* work clothes helmet wearing module
* mask wearing module

Content
========

* [Page display and introduction](#Page-display-and-introduction)
* [MatrixOne installation and startup of MySQL service](#MatrixOne-installation-and-startup-of-MySQL-service)
* [Install nodejs](#Install-nodejs)

## Page display and introduction

The front page is shown as follows:

![](./images/screen_example1.jpg)

After reading the image data from MatrixOne, the front page displays the following:

![](./images/screen_example2.jpg)

The specific information of the front-end page module is as follows:


1. Equipment quantity monitoring module

Displays the total number of devices, the number of operating devices and the number of abnormal devices in the smart park. 

❎It is currently a static module.


2. Vehicle monitoring module

Display the number of registered vehicles and existing vehicles in the park. The echart histogram component is used to display the remaining parking space capacity in the park. 

❎It is currently a static module.


3. Gender and age module

Displays the age and gender results recorded during face detection. Use the pie chart in the echart component to display the age and sex ratio of personnel respectively.

✅MatrixOne MySQL interaction mode: query the number of entries of different ages and genders in the face information table and return.


4. Pedestrian flow monitoring module

Display the number of people in the park today and the remaining number of people in the current park.

The lower part uses echart to display the broken line chart of people flow in the park over time, where the blue line shows the flow of people entering the park and the green line shows the flow of people leaving the park.

✅MatrixOne MySQL interaction mode: query the number of entries of people.


5. Alarm module

Display alarms for fireworks and abnormal manhole covers. Under normal conditions, the font is green, and the text content is "normal"; Under abnormal conditions, it is in red font and the text content is "alarm". Click the word "alarm" to display the image stored by the camera at the abnormal location in the middle of the screen.

✅MatrixOne MySQL interaction mode: For pyrotechnic alarm, query the last data in the pyrotechnic information table sorted by time and return it to the front-end page, read the string information stored in the environment field of the data, and truncate the string to read the confidence of pyrotechnic judgment. If the confidence is greater than 0.5, an alarm is displayed; For manhole cover alarms, query the manhole cover detection information table in chronological order. If the latest alarm data time is less than 2000 seconds, the alarm will be displayed on the front end.


6. Work clothes helmet wearing module

Display the fall detection alarm, which is the same as the operation logic of the alarm module. Use the echart histogram to display the wearing proportion of work clothes and helmets.

✅MatrixOne MySQL interaction mode: Query the fall detection table and return the last data sorted by time. If the time interval between this data and the current time is less than 2000 seconds, an alarm will be displayed. The work clothes and safety helmets are detected as static modules.


7. Mask wearing module

Use the echart Nightingale rose chart to display the wearing condition of the mask.

✅MatrixOne MySQL interaction mode: Query the face detection information table, return the number of entries with 0 and 1 masks, and transfer the number to the corresponding parameters of the echart component for display.

## MatrixOne installation and startup of MySQL service
### MatrixOne Installation

Use MatrixOne (stable version) code to build MatrixOne.

1. Get to use the 0.5.1 stable version branch.

```
git clone https://github.com/matrixorigin/matrixone.git
cd matrixone
git checkout 0.5.1
```

2. You can run make debug, make clean, or anything else our Makefile offers.

```
make config
make build
```

3. Launch MatrixOne server:

```
./mo-server system_vars_config.toml
```

see [MatrixOne](https://github.com/matrixorigin/matrixone) for more details.

###  Connecting to MatrixOne server

1. Install the MySQL client.

Oracle MySQL client needs to be installed.

```
sudo apt install mysql-server
```

2. Connect to MatrixOne server:

```
$ mysql -h IP -P PORT -uUsername -p
```

Use the built-in test account for example:

* user:dump
* password:111

```
$ mysql -h 127.0.0.1 -P 6001 -udump -p
Enter password:
```

see [MatrixOne](https://github.com/matrixorigin/matrixone) for more details.

## Install nodejs

In this case, the front and back end interaction is completed through nodejs. The js script file is divided into client side and server side. The client side corresponds to the front-end page and runs through the html page; The server is deployed on the server side and is used to connect to the matrixone MySQL database deployed on the server.

### 1. Install nodejs

This operation is performed on the server side.

```
sudo apt-get install nodejs
sudo apt-get install npm
```

### 2. Server:

Server code at: ./js/connect_matrixone_mysql_server.js

Place the code in the new folder on the server side and enter the folder.

Install the required modules:

```
npm install express
npm install mysql
```
start server:

```
node connect_matrixone_mysql_server.js
```

### 3. client:

Client code at ./js/connect_matrixone_mysql_client.js

When using, you need to modify the server IP address to the deployed server address.