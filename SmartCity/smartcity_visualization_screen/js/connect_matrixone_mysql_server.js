// 此代码在服务器端运行，即在运行MatrixOne MySQL数据库的机器运行

//1.首先 必须下载node.js 有node环境

//2.在你所选文件夹终端中执行 npm install mysql安装MySQL模块

// 下载express 执行 npm install express

// 下载MySQL 执行 npm install mysql

var express = require('express');
var ap_p = express();
ap_p.all('*', function(req, res, next) {             //设置跨域访问
     res.header("Access-Control-Allow-Origin", "*");
     res.header("Access-Control-Allow-Headers", "X-Requested-With");
     res.header("Access-Control-Allow-Methods","PUT,POST,GET,DELETE,OPTIONS");
     res.header("X-Powered-By",' 3.2.1');
     res.header("Content-Type", "application/json;charset=utf-8");
     next();
  });
ap_p.get('/falldown',function(req,res){           //配置接口api
    res.status(200);
    var mysql  = require('mysql');  
    console.log('mysql_falldown')
    var connection = mysql.createConnection({     
      host     : 'localhost',       
      user     : 'dump',              
      password : '111',       
      port: '6001',                   
      database: 'park' 
    }); 
    connection.connect();
    var  sql = 'SELECT * FROM falldown';
    connection.query(sql,function (err, result) {
            if(err){
              console.log('[SELECT ERROR] - ',err.message);
              return;
            }
           // re1=result[result.length-1];
           re1=result
           res.json(re1);
    });
    //connection.end();    
});
var server = ap_p.listen(3000,function(){
    var host = server.address().address;
    var port = server.address().port;
});
