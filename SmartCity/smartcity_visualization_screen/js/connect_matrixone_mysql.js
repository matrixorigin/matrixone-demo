// 引入MySQL 报错可尝试：

//1.首先 必须下载node.js 有node环境

//2.在你所选文件夹终端中执行 npm install mysql安装MySQL模块

var mysql  = require('mysql');
 
var connection = mysql.createConnection({
    host     : '192.168.194.94',
    user     : 'dump',
    password : '111',
    port: '6001',
    database: 'park'
});
 
connection.connect();
 
var  sql = 'SELECT * FROM face';
//查
connection.query(sql,function (err, result) {
    if(err){
        console.log('[SELECT ERROR] - ',err.message);
        return;
    }
 
    console.log('--------------------------SELECT----------------------------');
    console.log(result);
    console.log('------------------------------------------------------------\n\n');
});
 
connection.end();