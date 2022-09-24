//本案例对应数据库6个表，因此写了六个请求数据的函数。

//表名称smokefire, falldown

function mysql_read_falldown() {   
    b=$.ajax({
    type:'get', //请求方式
    url:'http://192.168.194.94:3000/falldown',  //请求地址,需要根据自己服务器ip地址来替换
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log(result)
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_falldown()", 10000);
}
mysql_read_falldown();